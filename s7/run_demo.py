#!/usr/bin/env python3
"""Single-command entrypoint. Regenerates submission_artifacts/ from nothing:

1. algebra_proof.py  -- training-free exactness proof for + - x / and the
   power/bilinear boundary result.
2. train_compare.py  -- trains dense / Kronecker-V1 / Kronecker-V2 embedding
   arms (3 seeds each) on even-only addition, evaluates in-distribution and
   two out-of-distribution axes: odd operand (parity, never seen in
   training) and magnitude (operand from a band larger than anything ever
   used as a training operand).
3. writes JSON + PNG artifacts and a run.log. No network calls, no manual steps.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import algebra_proof  # noqa: E402
from task import EVEN, MAGNITUDE_HIGH, NUM_RANGE, ODD  # noqa: E402
from train_compare import ARMS, run_comparison  # noqa: E402

ARTIFACTS = ROOT / "submission_artifacts"

ARM_LABELS = {
    "dense": "Dense table (course baseline)",
    "kron_v1": "Kronecker V1 (byte codec only)",
    "kron_v2": "Kronecker V2 (byte + meaning channel)",
}
ARM_COLORS = {"dense": "#999999", "kron_v1": "#4c72b0", "kron_v2": "#c44e52"}


def log(f, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    f.write(line + "\n")
    f.flush()


def aggregate_curves(runs: list[dict]) -> dict:
    steps = [p["step"] for p in runs[0]["curve"]]
    agg = {"steps": steps}
    for key in ("train_loss", "id_acc", "ood_acc", "magnitude_ood_acc"):
        arr = np.array([[p[key] for p in run["curve"]] for run in runs])
        agg[f"{key}_mean"] = arr.mean(axis=0).tolist()
        agg[f"{key}_std"] = arr.std(axis=0).tolist()
    return agg


def plot_accuracy_curves(curves: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    panels = (
        ("id_acc", "In-distribution (even + even)"),
        ("ood_acc", "Parity OOD (odd + odd, never trained on)"),
        ("magnitude_ood_acc", "Magnitude OOD (operand 200-298, never trained on)"),
    )
    for arm in ARMS:
        c = curves[arm]
        for ax, (key, _) in zip(axes, panels):
            mean = np.array(c[f"{key}_mean"])
            std = np.array(c[f"{key}_std"])
            ax.plot(c["steps"], mean, label=ARM_LABELS[arm], color=ARM_COLORS[arm])
            ax.fill_between(c["steps"], mean - std, mean + std, alpha=0.15, color=ARM_COLORS[arm])
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel("token accuracy")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradient_mass(grad_mass: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.2))
    plot_max = max(MAGNITUDE_HIGH) + 1
    xs = np.arange(plot_max)
    magnitude_set = set(MAGNITUDE_HIGH)
    colors = []
    for x in xs:
        if x < NUM_RANGE:
            colors.append("#4c72b0" if x % 2 == 0 else "#c44e52")
        elif x in magnitude_set:
            colors.append("#55a868")
        else:
            colors.append("#dddddd")
    ax.bar(xs, grad_mass[:plot_max], width=1.0, color=colors, linewidth=0)
    ax.set_title(
        f"Dense embedding table: accumulated per-row gradient mass, tokens 0-{plot_max - 1}\n"
        "blue=trained even  red=parity-OOD odd  green=magnitude-OOD (200-298)  "
        "gray=unsampled -- every non-blue bar is exactly zero",
        fontsize=10,
    )
    ax.set_xlabel("token id (= numeric value)")
    ax.set_ylabel("sum of ||grad|| over training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir(parents=True)

    with open(ARTIFACTS / "run.log", "w") as f:
        log(f, "run started")

        proof = algebra_proof.run_all()
        (ARTIFACTS / "algebra_proof.json").write_text(json.dumps(proof, indent=2))
        add_err = proof["additive_group"]["add_max_rel_err"]
        sub_err = proof["additive_group"]["sub_max_rel_err"]
        mul_err = proof["multiplicative_group"]["mul_max_rel_err"]
        div_err = proof["multiplicative_group"]["div_max_rel_err"]
        log(f, f"algebra proof: add_max_rel_err={add_err:.2e} sub_max_rel_err={sub_err:.2e}")
        log(f, f"algebra proof: mul_max_rel_err={mul_err:.2e} div_max_rel_err={div_err:.2e}")
        assert add_err < 1e-4 and sub_err < 1e-4, "additive group is not exact"
        assert mul_err < 1e-3 and div_err < 1e-3, "multiplicative group is not exact"
        log(f, "[PASS] algebra_proof_exact")

        boundary = proof["power_bilinear_boundary"]
        log(
            f, "power/bilinear boundary: outer_product_max_abs_err="
            f"{boundary['outer_product_max_abs_err']:.2e} "
            f"concat_err_over_outer_err={boundary['concat_err_over_outer_err']:.2e}"
        )
        assert boundary["concat_err_over_outer_err"] > 1e6
        log(f, "[PASS] power_needs_bilinear_readout")

        log(f, "training 3 arms x 3 seeds ...")
        _vocab, results = run_comparison(
            seeds=(0, 1, 2), n_steps=5000, batch_size=256, eval_every=500, eval_n=2000, lr=1.5e-3
        )
        log(f, "training complete")

        curves = {arm: aggregate_curves(results[arm]) for arm in ARMS}
        (ARTIFACTS / "accuracy_curves.json").write_text(json.dumps(curves, indent=2))

        final = {}
        for arm in ARMS:
            id_accs = [run["curve"][-1]["id_acc"] for run in results[arm]]
            ood_accs = [run["curve"][-1]["ood_acc"] for run in results[arm]]
            mag_accs = [run["curve"][-1]["magnitude_ood_acc"] for run in results[arm]]
            final[arm] = {
                "label": ARM_LABELS[arm],
                "num_params": results[arm][0]["num_params"],
                "final_id_acc_mean": float(np.mean(id_accs)),
                "final_id_acc_std": float(np.std(id_accs)),
                "final_ood_acc_mean": float(np.mean(ood_accs)),
                "final_ood_acc_std": float(np.std(ood_accs)),
                "final_magnitude_ood_acc_mean": float(np.mean(mag_accs)),
                "final_magnitude_ood_acc_std": float(np.std(mag_accs)),
            }
            log(
                f, f"[RESULT] {arm}: id_acc={final[arm]['final_id_acc_mean']:.3f} "
                f"ood_acc={final[arm]['final_ood_acc_mean']:.3f} "
                f"magnitude_ood_acc={final[arm]['final_magnitude_ood_acc_mean']:.3f} "
                f"params={final[arm]['num_params']}"
            )
        (ARTIFACTS / "final_results.json").write_text(json.dumps(final, indent=2))

        dense_run0 = results["dense"][0]
        grad_mass = np.array(dense_run0["grad_mass"])
        odd_mass = float(grad_mass[ODD].sum())
        even_mass = float(grad_mass[EVEN].sum())
        magnitude_mass = float(grad_mass[MAGNITUDE_HIGH].sum())
        gm = {
            "odd_total_grad_mass": odd_mass,
            "even_total_grad_mass": even_mass,
            "magnitude_high_total_grad_mass": magnitude_mass,
        }
        (ARTIFACTS / "gradient_mass.json").write_text(json.dumps(gm, indent=2))
        log(
            f, f"[RESULT] dense grad mass: even={even_mass:.1f} odd={odd_mass:.6f} "
            f"magnitude_high={magnitude_mass:.6f}"
        )
        assert odd_mass == 0.0, "odd-token rows received nonzero gradient"
        assert magnitude_mass == 0.0, "magnitude-band rows received nonzero gradient"
        log(f, "[PASS] odd_rows_never_received_gradient")
        log(f, "[PASS] magnitude_band_rows_never_received_gradient")

        plot_accuracy_curves(curves, ARTIFACTS / "accuracy_curves.png")
        log(f, "wrote accuracy_curves.png")
        plot_gradient_mass(grad_mass, ARTIFACTS / "gradient_mass.png")
        log(f, "wrote gradient_mass.png")

        log(f, "run complete")

    print(f"\nDone. See {ARTIFACTS}/")


if __name__ == "__main__":
    main()
