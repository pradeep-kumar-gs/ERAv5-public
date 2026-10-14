"""TinyStories -> byte-level BPE (8,192 tokens) -> uint16 token files (the same data as Session 13).

Every run reads the training tokens in the same order. A run that starts at token t
(a converted model continuing from a dense checkpoint) reads exactly the tokens the
dense run read from t on, so the two see identical data.
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN_TXT = DATA / "tinystories_train_head.txt"   # first 270 MB of TinyStoriesV2-GPT4-train.txt
VALID_TXT = DATA / "tinystories_valid.txt"
TOKENIZER = DATA / "bpe8192.json"
TRAIN_BIN = DATA / "train.bin"
VALID_BIN = DATA / "valid.bin"
VOCAB = 8192
EOT = "<|endoftext|>"


URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"


def download():
    """The first 270 MB of the training file (a byte range) and the whole validation file."""
    import urllib.request
    DATA.mkdir(exist_ok=True)
    for path, split, rng in [(TRAIN_TXT, "train", "bytes=0-269999999"), (VALID_TXT, "valid", None)]:
        if path.exists():
            continue
        req = urllib.request.Request(URL.format(split), headers={"Range": rng} if rng else {})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print(f"downloaded {path.name}: {path.stat().st_size:,} bytes")


def _stories(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = [s.strip() for s in text.split(EOT)]
    return [s for s in parts if s][:-1] if path == TRAIN_TXT else [s for s in parts if s]


def build(train_tokens=55_000_000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    stories = _stories(TRAIN_TXT)
    if not TOKENIZER.exists():
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(stories[:200_000], trainers.BpeTrainer(
            vocab_size=VOCAB, special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok.save(str(TOKENIZER))
    tok = Tokenizer.from_file(str(TOKENIZER))
    eot = tok.token_to_id(EOT)
    for path, out, limit in [(TRAIN_TXT, TRAIN_BIN, train_tokens), (VALID_TXT, VALID_BIN, 2_000_000)]:
        if out.exists():
            continue
        ids, n = [], 0
        src = _stories(path)
        for i in range(0, len(src), 10_000):
            for enc in tok.encode_batch(src[i:i + 10_000]):
                ids.append(np.asarray(enc.ids + [eot], dtype=np.uint16))
                n += len(enc.ids) + 1
            if n >= limit:
                break
        arr = np.concatenate(ids)[:limit]
        arr.tofile(out)
        print(f"{out.name}: {len(arr):,} tokens from {i + 10_000:,} stories")


def load(split):
    return np.memmap(TRAIN_BIN if split == "train" else VALID_BIN, dtype=np.uint16, mode="r")


class Stream:
    """Contiguous, non-overlapping (batch, seq+1) windows read in file order, from token `start`."""

    def __init__(self, batch, seq, split="train", start=0):
        self.tokens, self.batch, self.seq, self.pos = load(split), batch, seq, start

    def next(self):
        n = self.batch * self.seq + 1
        if self.pos + n > len(self.tokens):
            self.pos = 0
        chunk = torch.from_numpy(self.tokens[self.pos:self.pos + n].astype(np.int64))
        self.pos += n - 1
        x = chunk[:-1].view(self.batch, self.seq)
        y = chunk[1:].view(self.batch, self.seq)
        return x, y


if __name__ == "__main__":
    download()
    build()
