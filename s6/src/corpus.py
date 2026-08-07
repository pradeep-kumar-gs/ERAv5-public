"""Deterministic, fully offline synthetic corpus generator, plus a frozen
real-text seed for the code and agent lanes.

No network calls, no external files fetched at run time -- prose/eval are
built from fixed word banks and templates driven by a seeded RNG; code/agent
are read from corpus_seed/*.jsonl, a small (~70KB) slice of real commitpack
diffs and glaive tool-call transcripts committed straight into this repo
(see scripts/extract_corpus_seed.py for how it was derived, and the README
for why: it's real text, but small and static enough that Step 1 of grading
can regenerate byte-identical shards from a plain git clone, same guarantee
the fully-synthetic corpus gave). If corpus_seed/ is ever missing, code/agent
fall back to the synthetic generator below.

Four lanes, each existing specifically to force a different packing policy
in packing.py:
  - prose : plain paragraphs, concatenated + packed (baseline policy)
  - code  : real short commitpack diffs, document-isolated packing
  - agent : real glaive system/user/assistant/function-response transcripts,
            turn-level loss mask
  - eval  : held-out prose+code mix, never packed into a training batch
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_SEED_DIR = Path(__file__).resolve().parent.parent / "corpus_seed"


@dataclass
class Turn:
    role: str  # "system" | "user" | "assistant"
    text: str


@dataclass
class Document:
    doc_id: str
    lane: str
    text: str  # full concatenated text (for prose/code/eval)
    turns: list[Turn] = field(default_factory=list)  # populated for agent lane


SUBJECTS = ["the model", "a data engineer", "the scheduler", "the tokenizer",
            "the optimizer", "a shard", "the checkpoint", "the ledger",
            "the mixture planner", "a curriculum stage", "the packing policy",
            "an evaluation set", "the training loop", "a gradient", "the loss"]
VERBS = ["tracks", "reorders", "verifies", "consumes", "produces", "masks",
         "records", "compares", "reconstructs", "protects", "anneals",
         "samples", "reports", "validates", "replays"]
OBJECTS = ["the token span", "a batch id", "the attention pattern",
           "the loss mask", "a protected floor", "the offset counter",
           "the source hash", "a curriculum weight", "the position ids",
           "the rejection reason", "an audit record", "the shard manifest",
           "a golden proxy score", "the resume point", "the replay window"]
CONNECTORS = ["because", "so that", "even though", "after which",
              "provided that", "until", "while", "given that"]


def _sentence(rng: random.Random) -> str:
    s1 = rng.choice(SUBJECTS)
    v1 = rng.choice(VERBS)
    o1 = rng.choice(OBJECTS)
    conn = rng.choice(CONNECTORS)
    s2 = rng.choice(SUBJECTS)
    v2 = rng.choice(VERBS)
    o2 = rng.choice(OBJECTS)
    return f"{s1.capitalize()} {v1} {o1} {conn} {s2} {v2} {o2}."


def make_prose_docs(n_docs: int, sentences_per_doc: int, seed: int, lane: str = "prose") -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        text = " ".join(_sentence(rng) for _ in range(sentences_per_doc))
        docs.append(Document(doc_id=f"{lane}-{i:04d}", lane=lane, text=text))
    return docs


FUNC_NAMES = ["pack_batch", "verify_hash", "load_shard", "compile_mixture",
              "score_candidate", "advance_offset", "save_checkpoint",
              "replay_interval", "mask_loss", "resolve_lane", "fork_run",
              "audit_decision", "measure_throughput", "block_eval_shard"]
ARG_NAMES = ["shard_id", "offset", "lane", "stage", "threshold", "seed",
             "batch_id", "token_span", "weight", "floor"]
OPS = ["+", "-", "*", "//", "%"]


def _code_doc(rng: random.Random, doc_id: str) -> str:
    name = rng.choice(FUNC_NAMES)
    args = rng.sample(ARG_NAMES, k=3)
    body_lines = []
    for _ in range(rng.randint(3, 6)):
        a, b = rng.sample(args, k=2)
        op = rng.choice(OPS)
        var = rng.choice(ARG_NAMES)
        body_lines.append(f"    {var} = {a} {op} {b}")
    body_lines.append(f"    return {rng.choice(args)}")
    header = f"def {name}({', '.join(args)}):"
    return "\n".join([header, *body_lines])


def make_code_docs(n_docs: int, seed: int, lane: str = "code") -> list[Document]:
    rng = random.Random(seed)
    return [Document(doc_id=f"{lane}-{i:04d}", lane=lane, text=_code_doc(rng, f"{lane}-{i:04d}"))
            for i in range(n_docs)]


TOOLS = ["lookup_shard", "compute_hash", "query_ledger", "read_manifest", "score_batch"]
USER_ASKS = ["What is the current offset?", "Is this shard protected?",
             "Why was this batch rejected?", "What lane does this batch belong to?",
             "Can you verify the checkpoint hash?", "Which stage are we in?"]


def _agent_doc(rng: random.Random, doc_id: str) -> Document:
    system = "You are a data-system assistant. Use tools before answering."
    user = rng.choice(USER_ASKS)
    tool = rng.choice(TOOLS)
    arg = rng.choice(ARG_NAMES)
    tool_call = f"tool_call: {tool}({arg}=42)"
    answer = _sentence(rng)
    turns = [
        Turn("system", system),
        Turn("user", user),
        Turn("assistant", tool_call),
        Turn("assistant", answer),
    ]
    full_text = "\n".join(f"[{t.role}] {t.text}" for t in turns)
    return Document(doc_id=doc_id, lane="agent", text=full_text, turns=turns)


def make_agent_docs(n_docs: int, seed: int) -> list[Document]:
    rng = random.Random(seed)
    return [_agent_doc(rng, f"agent-{i:04d}") for i in range(n_docs)]


def _load_seed_jsonl(name: str) -> list[dict]:
    path = CORPUS_SEED_DIR / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_real_code_docs(lane: str = "code") -> list[Document]:
    return [Document(doc_id=r["doc_id"], lane=lane, text=r["text"])
            for r in _load_seed_jsonl("code_commitpack_sample.jsonl")]


def make_real_agent_docs() -> list[Document]:
    docs = []
    for r in _load_seed_jsonl("agent_glaive_sample.jsonl"):
        turns = [Turn(role=t["role"], text=t["text"]) for t in r["turns"]]
        full_text = "\n".join(f"[{t.role}] {t.text}" for t in turns)
        docs.append(Document(doc_id=r["doc_id"], lane="agent", text=full_text, turns=turns))
    return docs


def generate_corpus() -> dict[str, list[Document]]:
    """Fixed seeds (+ a frozen real-text seed for code/agent) -> byte-identical
    corpus on every call, every machine."""
    real_code = make_real_code_docs()
    real_agent = make_real_agent_docs()
    return {
        "prose": make_prose_docs(n_docs=40, sentences_per_doc=12, seed=101),
        "code": real_code if real_code else make_code_docs(n_docs=40, seed=202),
        "agent": real_agent if real_agent else make_agent_docs(n_docs=40, seed=303),
        "eval": (
            make_prose_docs(n_docs=8, sentences_per_doc=12, seed=404, lane="eval")
            + make_code_docs(n_docs=8, seed=505, lane="eval")
        ),
    }
