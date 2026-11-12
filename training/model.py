"""Pair ranker: a 2-layer, d=64 character transformer that scores (typo, candidate)."""
import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from keyboard import VOCAB_SIZE, MAX_LEN, MAX_WORD, CHARS, PAD

CONFIG = dict(version=1, d=64, heads=4, layers=2, ff=128, maxLen=MAX_LEN, maxWord=MAX_WORD, vocab=CHARS)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff); self.ff2 = nn.Linear(ff, d)

    def forward(self, x, key_mask):
        B, L, d = x.shape
        H, dh = self.heads, d // self.heads
        h = self.ln1(x)
        q = self.q(h).view(B, L, H, dh).transpose(1, 2)
        k = self.k(h).view(B, L, H, dh).transpose(1, 2)
        v = self.v(h).view(B, L, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / (dh ** 0.5)
        att = att.masked_fill(~key_mask[:, None, None, :], -1e9)
        att = att.softmax(-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.o(ctx)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class PairRanker(nn.Module):
    def __init__(self, cfg=CONFIG):
        super().__init__()
        d = cfg["d"]
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos_emb = nn.Embedding(cfg["maxLen"], d)
        self.seg_emb = nn.Embedding(2, d)
        self.layers = nn.ModuleList([Block(d, cfg["heads"], cfg["ff"]) for _ in range(cfg["layers"])])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, tokens, segs):
        """tokens, segs: [B, L] -> logits [B]"""
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)[None, :].expand(B, L)
        x = self.tok_emb(tokens) + self.pos_emb(pos) + self.seg_emb(segs)
        mask = tokens != PAD
        for layer in self.layers:
            x = layer(x, mask)
        x = self.lnf(x[:, 0])
        return self.head(x).squeeze(-1)


def export(model: PairRanker, out_dir: str):
    """Writes ranker.json (manifest) + ranker.bin (float32 little-endian) for the Swift side."""
    os.makedirs(out_dir, exist_ok=True)
    tensors, chunks, offset = [], [], 0
    for name, t in model.state_dict().items():
        arr = t.detach().cpu().float().numpy().astype("<f4").ravel()
        tensors.append({"name": name, "shape": list(t.shape), "offset": offset})
        chunks.append(arr); offset += arr.size
    with open(os.path.join(out_dir, "ranker.bin"), "wb") as f:
        f.write(np.concatenate(chunks).tobytes())
    with open(os.path.join(out_dir, "ranker.json"), "w") as f:
        json.dump({"config": model.cfg, "tensors": tensors}, f)
    return offset


def import_exported(model: PairRanker, manifest_path: str, bin_path: str):
    """Loads ranker.json + ranker.bin (the Swift-side format) into `model`."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    flat = np.fromfile(bin_path, dtype="<f4")
    state = {}
    for t in manifest["tensors"]:
        n = int(np.prod(t["shape"])) if t["shape"] else 1
        state[t["name"]] = torch.from_numpy(flat[t["offset"]:t["offset"] + n].reshape(t["shape"]).copy())
    model.load_state_dict(state)
    return model
