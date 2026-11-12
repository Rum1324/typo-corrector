"""Trains the decider: the small network that picks between keeping a typed word and replacing it.

  ../.build/release/typo-cli features --in .cache/data/instances/typed.tsv \
      --hidden .cache/data/instances/hidden.npy --out .cache/data/decider \
      --context-dir runs/lm --ranker-dir runs/pair_v2
  .venv/bin/python -I decider.py [--out runs/decider]

The option rows (one per candidate, plus "keep") are written by the Swift engine itself, so the
network sees in training exactly the numbers it will see in the app. Each word is one list; the
network scores every row and a softmax over the list is trained to pick the meant word, or "keep"
when the word was typed as meant or the meant word is not among the candidates.
Runs on the CPU in a few minutes. Writes context.json (read by Sources/TypoCore/ContextModel.swift).
"""
import argparse, json, os, sys, time, zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / ".cache" / "data" / "decider"
TYPED = HERE / ".cache" / "data" / "instances" / "typed.tsv"
K = 16                                   # keep + up to 15 candidates


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(data_dir, typed_path):
    columns = (data_dir / "columns.txt").read_text().split()
    meant = {}
    with open(typed_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            meant[c[0]] = (c[1], c[6], c[7], c[8])
    feats = np.memmap(data_dir / "features.f32", dtype="<f4", mode="r").reshape(-1, len(columns))
    with open(data_dir / "rows.tsv", encoding="utf-8") as f:
        table = [line.rstrip("\n").split("\t") for line in f]
    n = sum(1 for row in table if row[2])
    X = np.zeros((n, K, len(columns)), dtype=np.float32)       # one copy: 450k words are 0.9 GB
    M = np.zeros((n, K), dtype=bool)
    Y, kind, val, at, i = [], [], [], 0, 0
    skipped = {}
    if True:
        for ident, note, words in table:
            words = words.split(" ") if words else []
            if not words:
                skipped[note.split(":")[0]] = skipped.get(note.split(":")[0], 0) + 1
                continue
            rows = feats[at:at + len(words)]
            at += len(words)
            typed, word, source, src_line = meant[ident]
            low = word.lower()
            if typed.lower() == low:
                y, k = 0, 0                                  # typed as meant
            elif low in words[1:]:
                y, k = words.index(low, 1), 1                # a typo the engine can fix
            else:
                y, k = 0, 2                                  # a typo whose fix is not on offer: hands off
            X[i, :len(words)] = rows[:K]
            M[i, :len(words)] = True
            i += 1
            Y.append(y); kind.append(k)
            val.append(zlib.crc32(f"{source}:{src_line}".encode()) % 10 == 0)   # whole lines are held out
    assert at == len(feats), "rows.tsv and features.f32 disagree"
    log(f"{len(X)} words with options; no options for {skipped}")
    return columns, torch.from_numpy(X), torch.from_numpy(M), torch.tensor(Y), torch.tensor(kind), torch.tensor(val)


class Net(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        sizes = [n] + hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x.squeeze(-1)


def report(logits, y, kind, known, name):
    """Outcomes at several probability thresholds. A replacement is made when the best option is
    not "keep" and its probability reaches the threshold."""
    p = logits.softmax(-1)
    conf, pick = p.max(-1)
    typo, clean = kind > 0, kind == 0
    log(f"{name}: {int(typo.sum())} typos ({int((kind == 1).sum())} fixable), {int(clean.sum())} clean words")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        acts = (pick != 0) & (conf >= t)
        fixed = (acts & (pick == y) & (kind == 1)).sum().item()
        wrong = (acts & typo & ~((pick == y) & (kind == 1))).sum().item()
        false = (acts & clean).sum().item()
        kf = (acts & clean & known).sum().item()
        rw = (typo & known)
        rw_fixed = (acts & (pick == y) & (kind == 1) & known).sum().item()
        log(f"  p>={t:.2f}: fixed {fixed / max(typo.sum().item(), 1):6.1%}  wrong {wrong / max(typo.sum().item(), 1):5.1%}  "
            f"false fixes {false / max(clean.sum().item(), 1):7.3%} ({kf} on dictionary words)  "
            f"real-word typos fixed {rw_fixed / max(rw.sum().item(), 1):6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--typed", default=str(TYPED))
    ap.add_argument("--out", default=str(HERE / "runs" / "decider"))
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 32])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--clean-weight", type=float, default=2.0, help="loss weight of a word typed as meant")
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to blank out (ablation)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--members", type=int, default=1, help="networks trained from different random starts and averaged")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    columns, X, M, Y, kind, val = load(Path(args.data), Path(args.typed))
    known = X[:, 0, columns.index("typedKnown")] > 0.5
    flat = X[M]
    mean, scale = flat.mean(0), flat.std(0).clamp(min=1e-3)
    for name in args.drop:                                   # a blanked column is constant after scaling
        X[:, :, columns.index(name)] = mean[columns.index(name)]
    Xs = X.sub_(mean).div_(scale)                            # in place: X is not used again
    tr, va = ~val, val
    weight = torch.where(kind == 0, torch.tensor(args.clean_weight), torch.tensor(1.0))
    idx = tr.nonzero().squeeze(1)
    steps = args.epochs * ((len(idx) + args.batch - 1) // args.batch)
    models, out = [], 0
    for member in range(args.members):
      torch.manual_seed(args.seed + member)
      model = Net(len(columns), args.hidden)
      models.append(model)
      opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
      sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
      log(f"network {member + 1}/{args.members}: {len(idx)} training words, {int(va.sum())} held out, "
          f"{sum(p.numel() for p in model.parameters())} parameters")

      def logits_of(x, m):
          return model(x).masked_fill(~m, -1e4)

      for ep in range(args.epochs):
        model.train()
        perm = idx[torch.randperm(len(idx))]
        total, seen = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = (F.cross_entropy(logits_of(Xs[b], M[b]), Y[b], reduction="none") * weight[b]).sum() / weight[b].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            total += loss.item() * len(b); seen += len(b)
        model.eval()
        if ep + 1 == args.epochs or (ep + 1) % 5 == 0:
            with torch.no_grad():
                v = F.cross_entropy(logits_of(Xs[va], M[va]), Y[va]).item()
            log(f"epoch {ep + 1}/{args.epochs} loss {total / seen:.4f} held-out {v:.4f}")
      with torch.no_grad():
          out = out + logits_of(Xs[va], M[va]) / args.members       # the app averages the networks' scores
    log(f"held-out loss of the average: {F.cross_entropy(out, Y[va]).item():.4f}")
    report(out, Y[va], kind[va], known[va], "held-out lines")

    os.makedirs(args.out, exist_ok=True)
    dense = lambda net: [{"w": l.weight.detach().tolist(), "b": l.bias.detach().tolist()} for l in net.layers]
    spec = {"version": 1, "features": columns, "mean": mean.tolist(), "scale": scale.tolist(),
            "layers": dense(models[0]), "ensemble": [dense(net) for net in models[1:]]}
    with open(Path(args.out) / "context.json", "w") as f:
        json.dump(spec, f)
    torch.save({"models": [net.state_dict() for net in models], "args": vars(args)}, Path(args.out) / "decider.pt")
    log(f"wrote {Path(args.out) / 'context.json'}")


if __name__ == "__main__":
    main()
