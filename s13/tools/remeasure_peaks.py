"""The main runs were trained before the peak sampler was fixed (module-boundary hooks
missed the loss transients). Re-measure each run's step peak with the per-op sampler
for its exact config; keep the old figure under peak_bytes_module_hooks."""
import json
from pathlib import Path

from rev.train import RunConfig, measure_peak

for p in sorted(Path("results").glob("*.json")):
    log = json.loads(p.read_text())
    if "config" not in log or "peak_bytes_module_hooks" in log:
        continue
    cfg = RunConfig(**{k: v for k, v in log["config"].items() if k != "notes"})
    log["peak_bytes_module_hooks"] = log["peak_bytes"]
    log["peak_bytes"] = measure_peak(cfg)
    for row in log["train"]:
        row.pop("peak_bytes", None)
    p.write_text(json.dumps(log, indent=1))
    print(f"{p.name}: {log['peak_bytes_module_hooks']/2**30:.2f} -> {log['peak_bytes']/2**30:.2f} GiB", flush=True)
