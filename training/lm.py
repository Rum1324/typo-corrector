"""Context language model: which word fits after the words typed so far on this line?

A small causal word transformer over the tok.py vocabulary (the 80k lexicon). Embeddings are
frequency-banded and tied with the output layer (frequent words get 128 dimensions, rare ones 32),
which keeps the model small and the softmax cheap. Inference is re-implemented in Swift
(Sources/TypoCore/ContextLM.swift) from the exported lm.json + lm.bin.

  .venv/bin/python -I lm.py --bench                 throughput of one configuration (random tokens)
  .venv/bin/python -I lm.py --train --steps 60000   train on .cache/data/corpus, checkpoint, export
  .venv/bin/python -I lm.py --eval                  validation perplexity per source
"""
import argparse, json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tok import Vocab, BOS, CUT, N_SPECIAL

HERE = Path(__file__).resolve().parent
CORPUS = HERE / ".cache" / "data" / "corpus"
RUN = HERE / "runs" / "lm"

# bands: vocabulary ids below 10016 use 128-d embeddings, below 40016 64-d, the rest 32-d.
CONFIG = dict(version=1, d=256, heads=4, layers=4, ff=1024, maxLen=32, bands=[10_000 + N_SPECIAL, 40_000 + N_SPECIAL], dims=[128, 64, 32])


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff); self.ff2 = nn.Linear(ff, d)

    def forward(self, x, mask):
        B, L, d = x.shape
        H = self.heads
        h = self.ln1(x)
        q = self.q(h).view(B, L, H, d // H).transpose(1, 2)
        k = self.k(h).view(B, L, H, d // H).transpose(1, 2)
        v = self.v(h).view(B, L, H, d // H).transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(ctx.transpose(1, 2).reshape(B, L, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class ContextLM(nn.Module):
    def __init__(self, vocab_size, cfg=CONFIG):
        super().__init__()
        self.cfg = dict(cfg, vocab=vocab_size)
        d, dims = cfg["d"], cfg["dims"]
        self.bounds = [0] + list(cfg["bands"]) + [vocab_size]
        sizes = [self.bounds[i + 1] - self.bounds[i] for i in range(len(dims))]
        self.emb = nn.ModuleList([nn.Embedding(n, e) for n, e in zip(sizes, dims)])
        self.in_proj = nn.ModuleList([nn.Linear(e, d, bias=False) for e in dims])
        self.out_proj = nn.ModuleList([nn.Linear(d, e, bias=False) for e in dims])
        self.out_bias = nn.ParameterList([nn.Parameter(torch.zeros(n)) for n in sizes])
        # The head predicts a frequent word or "one of band i"; band i then picks the word.
        self.cluster = nn.Parameter(torch.randn(len(dims) - 1, dims[0]) * 0.02)
        self.cluster_bias = nn.Parameter(torch.zeros(len(dims) - 1))
        self.pos_emb = nn.Embedding(cfg["maxLen"], d)
        self.layers = nn.ModuleList([Block(d, cfg["heads"], cfg["ff"]) for _ in range(cfg["layers"])])
        self.lnf = nn.LayerNorm(d)
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def embed(self, tokens):
        x = 0
        for i, (emb, proj) in enumerate(zip(self.emb, self.in_proj)):
            lo, hi = self.bounds[i], self.bounds[i + 1]
            inside = (tokens >= lo) & (tokens < hi)
            e = proj(emb((tokens - lo).clamp(0, hi - lo - 1)))
            x = x + e * inside.unsqueeze(-1)
        return x

    @staticmethod
    def layout(tokens):
        """Lines are independent: positions restart at <s>, attention stays inside the line."""
        B, L = tokens.shape
        idx = torch.arange(L, device=tokens.device).expand(B, L)
        starts = tokens == BOS
        seg = starts.cumsum(1)
        start_idx = torch.where(starts, idx, torch.zeros_like(idx)).cummax(1).values
        pos = idx - start_idx
        mask = (seg[:, :, None] == seg[:, None, :]) & (idx[:, None, :] <= idx[:, :, None])
        return pos, mask[:, None]

    def hidden(self, tokens):
        """tokens [B, L] -> [B, L, d]: the state after each token, used to predict the next one."""
        pos, mask = self.layout(tokens)
        x = self.embed(tokens) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lnf(x)

    def head_log_probs(self, h):
        w = torch.cat([self.emb[0].weight, self.cluster])
        b = torch.cat([self.out_bias[0], self.cluster_bias])
        return F.log_softmax(self.out_proj[0](h) @ w.T + b, dim=-1)

    def nll(self, h, targets):
        """h [N, d], targets [N] -> negative log-likelihood [N]."""
        n0 = self.bounds[1]
        logp = self.head_log_probs(h)
        cuts = torch.tensor(self.bounds[1:-1], device=targets.device)
        band = torch.bucketize(targets, cuts, right=True)
        head_target = torch.where(band == 0, targets, n0 + band - 1)
        nll = -logp.gather(1, head_target[:, None]).squeeze(1)
        for i in range(1, len(self.emb)):
            idx = (band == i).nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            lp = F.log_softmax(self.out_proj[i](h[idx]) @ self.emb[i].weight.T + self.out_bias[i], dim=-1)
            nll = nll.index_add(0, idx, -lp.gather(1, (targets[idx] - self.bounds[i])[:, None]).squeeze(1))
        return nll

    def head_nll(self, h, targets):
        """h [N, d], targets [N] -> [N]: -log P(the word, or its band when the word is rare)."""
        cuts = torch.tensor(self.bounds[1:-1], device=targets.device)
        band = torch.bucketize(targets, cuts, right=True)
        head_target = torch.where(band == 0, targets, self.bounds[1] + band - 1)
        return -self.head_log_probs(h).gather(1, head_target[:, None]).squeeze(1)

    def tail_nll(self, i, h, targets):
        """h [K, d], targets [K] -> [K]: -log P(word | band i). Targets outside the band are clamped."""
        lp = F.log_softmax(self.out_proj[i](h) @ self.emb[i].weight.T + self.out_bias[i], dim=-1)
        inside = (targets - self.bounds[i]).clamp(0, self.bounds[i + 1] - self.bounds[i] - 1)
        return -lp.gather(1, inside[:, None]).squeeze(1)

    def train_loss(self, h, targets, valid, tails):
        """Mean NLL with fixed tensor shapes (MPS compiles and caches a graph for every new shape).
        h [N, d], targets [N], valid [N] float; tails: per rare band (positions [K], weights [K])."""
        total = (self.head_nll(h, targets) * valid).sum()
        for i, (idx, weight) in enumerate(tails, start=1):
            total = total + (self.tail_nll(i, h[idx], targets[idx]) * weight).sum()
        return total / valid.sum().clamp(min=1)

    @torch.no_grad()
    def log_probs(self, h, words):
        """h [d], words [K] (token ids) -> log P(word | context) [K]. Reference for the Swift port."""
        logp0 = self.head_log_probs(h[None])[0]
        out = torch.empty(len(words), device=h.device)
        tails = {}
        for j, w in enumerate(words.tolist()):
            band = sum(w >= b for b in self.bounds[1:-1])
            if band == 0:
                out[j] = logp0[w]
                continue
            if band not in tails:
                tails[band] = F.log_softmax(self.out_proj[band](h) @ self.emb[band].weight.T + self.out_bias[band], dim=-1)
            out[j] = logp0[self.bounds[1] + band - 1] + tails[band][w - self.bounds[band]]
        return out


def export(model, out_dir):
    """lm.json (manifest) + lm.bin (float32 little-endian), the layout NeuralRanker already reads."""
    os.makedirs(out_dir, exist_ok=True)
    tensors, chunks, offset = [], [], 0
    for name, t in model.state_dict().items():
        arr = t.detach().cpu().float().numpy().astype("<f4").ravel()
        tensors.append({"name": name, "shape": list(t.shape), "offset": offset})
        chunks.append(arr); offset += arr.size
    with open(os.path.join(out_dir, "lm.bin"), "wb") as f:
        f.write(np.concatenate(chunks).tobytes())
    with open(os.path.join(out_dir, "lm.json"), "w") as f:
        json.dump({"config": model.cfg, "tensors": tensors}, f)
    return offset


def load_checkpoint(path=None, dev=None):
    ck = torch.load(path or RUN / "ckpt.pt", map_location="cpu")
    model = ContextLM(ck["cfg"]["vocab"], {k: v for k, v in ck["cfg"].items() if k != "vocab"})
    model.load_state_dict(ck["model"])
    return model.to(dev or device()).eval(), ck


# ---------------------------------------------------------------- data

def load_split(split):
    parts = {}
    for path in sorted(CORPUS.glob(f"*.{split}.u32")):
        parts[path.name.split(".")[0]] = np.fromfile(path, dtype="<u4")
    return parts


class Sampler:
    """Random windows of L+1 tokens from all sources (proportional to their size)."""

    # Share of positions given a slot in each rare band's softmax (real text: about 6% and 2%).
    TAIL_SLOTS = (0.08, 0.035)

    def __init__(self, parts, L, seed, bounds):
        self.data = np.empty(sum(len(v) for v in parts.values()), dtype="<u4")
        at = 0
        for name in list(parts):          # one copy in memory: each part is freed once copied
            v = parts.pop(name)
            self.data[at:at + len(v)] = v
            at += len(v)
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.ar = np.arange(L + 1)
        self.bounds = bounds

    def batch(self, B, dev):
        """inputs [B, L], targets [B*L], valid [B*L], tails [(positions [K], weights [K])]."""
        offs = self.rng.integers(0, len(self.data) - self.L - 1, size=B)
        rows = self.data[offs[:, None] + self.ar].astype(np.int64)
        x, y = rows[:, :-1].copy(), rows[:, 1:].reshape(-1)
        x[x[:, 0] != BOS, 0] = CUT        # the window starts inside a line: what came before is unknown
        valid = y != BOS                  # the end of a line is not predicted
        tails = []
        for i, share in enumerate(self.TAIL_SLOTS, start=1):
            k = int(len(y) * share)
            pos = np.flatnonzero(valid & (y >= self.bounds[i]) & (y < self.bounds[i + 1]))
            scale = 1.0
            if len(pos) > k:              # rare: keep k of them and reweight
                scale = len(pos) / k
                pos = self.rng.choice(pos, size=k, replace=False)
            idx = np.zeros(k, dtype=np.int64); w = np.zeros(k, dtype=np.float32)
            idx[:len(pos)] = pos; w[:len(pos)] = scale
            tails.append((torch.from_numpy(idx).to(dev), torch.from_numpy(w).to(dev)))
        return (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                torch.from_numpy(valid.astype(np.float32)).to(dev), tails)

    def loss(self, model, B, dev):
        x, y, valid, tails = self.batch(B, dev)
        h = model.hidden(x)
        return model.train_loss(h.reshape(-1, h.shape[-1]), y, valid, tails)


@torch.no_grad()
def perplexity(model, tokens, dev, L, batch=128, max_tokens=120_000, chunk=512):
    """Validation perplexity. Every tensor has a fixed shape: on MPS each new shape adds a compiled
    graph to a cache that is never freed, and a pass over ragged batches can exhaust memory."""
    model.eval()
    n = (min(len(tokens), max_tokens) - 1) // L
    n -= n % batch
    b, d = model.bounds, model.cfg["d"]
    tot, cnt = 0.0, 0
    for at in range(0, n, batch):
        rows = np.stack([tokens[i * L:i * L + L + 1] for i in range(at, at + batch)]).astype(np.int64)
        x, y = rows[:, :-1].copy(), rows[:, 1:].reshape(-1)
        x[x[:, 0] != BOS, 0] = CUT
        valid = y != BOS
        h = model.hidden(torch.from_numpy(x).to(dev)).reshape(-1, d)
        yt = torch.from_numpy(y).to(dev)
        tot += (model.head_nll(h, yt) * torch.from_numpy(valid.astype(np.float32)).to(dev)).sum().item()
        for i in range(1, len(model.emb)):
            pos = np.flatnonzero(valid & (y >= b[i]) & (y < b[i + 1]))
            for c in range(0, len(pos), chunk):
                idx = np.zeros(chunk, dtype=np.int64); w = np.zeros(chunk, dtype=np.float32)
                part = pos[c:c + chunk]
                idx[:len(part)] = part; w[:len(part)] = 1
                it = torch.from_numpy(idx).to(dev)
                tot += (model.tail_nll(i, h[it], yt[it]) * torch.from_numpy(w).to(dev)).sum().item()
        cnt += int(valid.sum())
    model.train()
    return math.exp(tot / max(cnt, 1))


def log(*a, file=None):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--d", type=int, default=CONFIG["d"])
    ap.add_argument("--layers", type=int, default=CONFIG["layers"])
    ap.add_argument("--ff", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--minutes", type=float, default=0, help="stop (checkpointed) after this long; rerun with --resume")
    ap.add_argument("--duty", type=float, default=1.0, help="share of the time the GPU works (0.3 = idle 70%% of the time)")
    ap.add_argument("--mem-fraction", type=float, default=0.3, help="cap on MPS memory: fail instead of starving the system")
    ap.add_argument("--run", default=str(RUN))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    if dev.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mem_fraction)
    cfg = dict(CONFIG, d=args.d, layers=args.layers, ff=args.ff or 4 * args.d)
    L = cfg["maxLen"]
    vocab_size = Vocab().size
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    logfile = run / "train.log"

    if args.bench:
        model = ContextLM(vocab_size, cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Band mix of real text: about 92% of tokens in the head band, 6% in the middle, 2% rare.
        rng = np.random.default_rng(0)
        r, b = rng.random(4_000_000), model.bounds
        ids = np.where(r < 0.92, rng.integers(N_SPECIAL, b[1], r.size),
                       np.where(r < 0.98, rng.integers(b[1], b[2], r.size), rng.integers(b[2], b[3], r.size)))
        ids[rng.random(r.size) < 0.05] = BOS
        fake = {"bench": ids.astype("<u4")}
        sampler = Sampler(fake, L, 0, model.bounds)
        print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  d={cfg['d']} layers={cfg['layers']} batch={args.batch} device={dev}")
        for step in range(45):
            if step == 5:
                torch.mps.synchronize() if dev.type == "mps" else None
                t0 = time.time()
            loss = sampler.loss(model, args.batch, dev)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss.item()
        dt = (time.time() - t0) / 40
        print(f"{dt * 1000:.0f} ms/step, {args.batch * L / dt / 1000:.1f}k tokens/s, {args.batch * L * 3600 / dt / 1e6:.0f}M tokens/hour")
        return

    if args.eval:
        model, ck = load_checkpoint(run / "ckpt.pt", dev)
        for name, toks in load_split("val").items():
            print(f"{name}: ppl {perplexity(model, toks, dev, L):.1f}")
        return

    if args.train:
        train = load_split("train")
        val = load_split("val")
        if not args.resume or not (run / "ckpt.pt").exists():
            log("sources", {k: f"{len(v) / 1e6:.1f}M" for k, v in train.items()}, file=logfile)
        model = ContextLM(vocab_size, cfg).to(dev)
        sampler = Sampler(train, L, args.seed, model.bounds)   # takes the arrays out of `train`
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith(("emb.", "pos_emb"))]
        rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith(("emb.", "pos_emb")))]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                                lr=args.lr, betas=(0.9, 0.98))
        start = 0
        if args.resume and (run / "ckpt.pt").exists():
            ck = torch.load(run / "ckpt.pt", map_location="cpu")
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
            sampler.rng = np.random.default_rng(args.seed + start)
            log(f"resumed at step {start}", file=logfile)
        log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M cfg {cfg} steps {args.steps} batch {args.batch} lr {args.lr}", file=logfile)

        def lr_at(step):
            if step < args.warmup:
                return args.lr * (step + 1) / args.warmup
            t = (step - args.warmup) / max(1, args.steps - args.warmup)
            return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

        def save(step):
            tmp = run / "ckpt.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": model.cfg}, tmp)
            os.replace(tmp, run / "ckpt.pt")

        model.train()
        started = t0 = time.time()
        run_loss, seen, step = 0.0, 0, start
        for step in range(start, args.steps):
            tick = time.time()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = sampler.loss(model, args.batch, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if args.duty < 1 or (step + 1) % 50 == 0:
                run_loss += loss.item(); seen += 1          # .item() waits for the GPU
            if args.duty < 1:
                time.sleep((time.time() - tick) * (1 / max(args.duty, 0.05) - 1))
            if (step + 1) % 500 == 0:
                rate = 500 * args.batch * L / (time.time() - t0)
                log(f"step {step + 1}/{args.steps} loss {run_loss / max(seen, 1):.3f} lr {lr_at(step):.2e} {rate / 1000:.1f}k tok/s", file=logfile)
                t0, run_loss, seen = time.time(), 0.0, 0
            done = step + 1 == args.steps
            out_of_time = args.minutes > 0 and time.time() - started > args.minutes * 60
            if (step + 1) % args.save_every == 0 or done or out_of_time:
                save(step + 1)
            if (step + 1) % args.eval_every == 0 or done:
                ppl = {k: round(perplexity(model, v, dev, L), 1) for k, v in val.items()}
                log(f"step {step + 1} val ppl {ppl}", file=logfile)
                t0 = time.time()
            if out_of_time and not done:
                log(f"paused at step {step + 1}/{args.steps} after {args.minutes:.0f} min; rerun with --resume", file=logfile)
                return 3
        n = export(model, run)
        log(f"exported {n} floats to {run}/lm.bin", file=logfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
