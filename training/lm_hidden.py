"""Runs the context language model over the sampled typing instances (build_instances.py).

  .venv/bin/python -I lm_hidden.py [--run runs/lm] [--limit N]

Writes .cache/data/instances/hidden.npy: float16 [N, d], the model state after each instance's
context. Word probabilities for any candidate list are computed from it later on the CPU, so the
GPU is needed only once per language model. Resumable: finished chunks are kept on disk.
"""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
import torch
import lm

HERE = Path(__file__).resolve().parent
INST = HERE / ".cache" / "data" / "instances"
BATCH = 1024
CHUNK = 100 * BATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(lm.RUN))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    args = ap.parse_args()
    dev = lm.device()
    if dev.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mem_fraction)
    model, ck = lm.load_checkpoint(Path(args.run) / "ckpt.pt", dev)
    ctx = np.load(INST / "ctx.npy")
    if args.limit:
        ctx = ctx[:args.limit]
    n, d = len(ctx), model.cfg["d"]
    parts = INST / "hidden_parts"
    parts.mkdir(exist_ok=True)
    stamp = f"step{ck['step']}"
    t0 = time.time()
    for c0 in range(0, n, CHUNK):
        path = parts / f"{stamp}_{c0}.npy"
        if path.exists():
            continue
        block = ctx[c0:c0 + CHUNK]
        out = np.empty((len(block), d), dtype=np.float16)
        for b0 in range(0, len(block), BATCH):
            rows = block[b0:b0 + BATCH]
            m = len(rows)
            if m < BATCH:                                   # pad the last batch: one tensor shape only
                rows = np.concatenate([rows, np.repeat(rows[:1], BATCH - m, axis=0)])
            x = torch.from_numpy(rows.astype(np.int64)).to(dev)
            last = (x != 0).sum(1) - 1
            with torch.no_grad():
                h = model.hidden(x)[torch.arange(BATCH, device=dev), last]
            out[b0:b0 + m] = h[:m].float().cpu().numpy().astype(np.float16)
        np.save(path, out)
        print(f"{time.strftime('%H:%M:%S')} {min(c0 + CHUNK, n)}/{n} contexts, {time.time() - t0:.0f}s", flush=True)
    np.save(INST / "hidden.npy", np.concatenate([np.load(parts / f"{stamp}_{c0}.npy") for c0 in range(0, n, CHUNK)]))
    print(f"wrote hidden.npy [{n}, {d}] from checkpoint step {ck['step']}")


if __name__ == "__main__":
    main()
