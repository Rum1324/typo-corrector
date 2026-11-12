"""Train the pair ranker.

  python train.py --base                      train from synthetic typos, export to ../Resources
  python train.py --base --realistic --out D  train on typos drawn from the fitted error model
                                              (noise.py) plus the real typo tables; export to D
  python train.py --personal --db DB --out DIR  fine-tune the base ranker on your own fix pairs

The personal run is what the app launches when the Mac is idle.
"""
import argparse, os, random, sqlite3, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import torch
import torch.nn.functional as F
from keyboard import encode_pair, MAX_LEN
from synth import make_typo, load_lexicon, sample_words
from candidates import Candidates
from model import PairRanker, export, import_exported, CONFIG

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "Resources"
CACHE = Path(__file__).resolve().parent / ".cache"
K = 8  # candidates per list (1 positive + up to 7 negatives)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_lists(pairs, cands, rng, negatives_only=False):
    """pairs: list of (typo, correct). Returns tokens [N,K,L], segs [N,K,L], valid [N,K], label [N].
    For negatives_only (undo events) the 'correct' word is the one to push DOWN: label=-1."""
    T, S, V, Y = [], [], [], []
    done = [[], [], [], []]

    def flush():                                    # bytes, not Python ints: 700k lists stay under 1 GB
        if T:
            for part, rows, kind in zip(done, (T, S, V, Y), (torch.uint8, torch.uint8, torch.bool, torch.long)):
                part.append(torch.tensor(rows, dtype=kind))
            T.clear(); S.clear(); V.clear(); Y.clear()

    for typo, correct in pairs:
        if len(T) >= 20_000:
            flush()
        others = [c for c in cands.lookup(typo) if c != correct]
        if negatives_only:
            if not others:
                continue
            lst = [correct] + others[:K - 1]
            label = -1
        else:
            if not others:
                others = [c for c in cands.lookup(typo, max_dist=2, top_k=K) if c != correct]
            lst = [correct] + others[:K - 1]
            if len(lst) < 2:
                continue
            label = 0
        toks, segs = zip(*[encode_pair(typo, c) for c in lst])
        toks, segs = list(toks), list(segs)
        valid = [1] * len(lst)
        dummy_t, dummy_s = encode_pair("", "")
        while len(toks) < K:
            toks.append(dummy_t); segs.append(dummy_s); valid.append(0)
        T.append(toks); S.append(segs); V.append(valid); Y.append(label)
    flush()
    if not done[0]:
        return (torch.zeros((0, K, MAX_LEN), dtype=torch.uint8), torch.zeros((0, K, MAX_LEN), dtype=torch.uint8),
                torch.zeros((0, K), dtype=torch.bool), torch.zeros((0,), dtype=torch.long))
    return tuple(torch.cat(part) for part in done)


def list_loss(model, toks, segs, valid, labels):
    B, Kk, L = toks.shape
    logits = model(toks.view(B * Kk, L), segs.view(B * Kk, L)).view(B, Kk)
    logits = logits.masked_fill(~valid, -1e4)
    pos = labels >= 0
    loss = torch.zeros((), device=toks.device)
    if pos.any():
        loss = loss + F.cross_entropy(logits[pos], labels[pos])
    neg = ~pos
    if neg.any():
        # push the rejected candidate (index 0) down: -log(1 - p0)
        p0 = logits[neg].log_softmax(-1)[:, 0]
        loss = loss + (-(torch.log1p(-p0.exp().clamp(max=1 - 1e-6)))).mean()
    acc = ((logits[pos].argmax(-1) == labels[pos]).float().mean().item()) if pos.any() else float("nan")
    return loss, acc


def run_epochs(model, data, epochs, lr, batch, dev, replay=None, replay_ratio=0):
    toks, segs, valid, y = data
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = toks.shape[0]
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, accs, nb = 0.0, [], 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            bt, bs, bv, by = toks[idx], segs[idx], valid[idx], y[idx]
            if replay is not None and replay_ratio > 0:
                rt, rs, rv, ry = replay
                ridx = torch.randint(0, rt.shape[0], (min(len(idx) * replay_ratio, rt.shape[0]),))
                bt = torch.cat([bt, rt[ridx]]); bs = torch.cat([bs, rs[ridx]])
                bv = torch.cat([bv, rv[ridx]]); by = torch.cat([by, ry[ridx]])
            loss, acc = list_loss(model, bt.to(dev).long(), bs.to(dev).long(), bv.to(dev), by.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
            if acc == acc:
                accs.append(acc)
        log(f"epoch {ep + 1}/{epochs} loss {tot / max(nb, 1):.4f} top1 {sum(accs) / max(len(accs), 1):.3f}")


@torch.no_grad()
def evaluate(model, data, dev, batch=1024):
    model.eval()
    toks, segs, valid, y = data
    correct, total = 0, 0
    for i in range(0, toks.shape[0], batch):
        bt, bs = toks[i:i + batch].to(dev).long(), segs[i:i + batch].to(dev).long()
        bv, by = valid[i:i + batch].to(dev), y[i:i + batch].to(dev)
        B, Kk, L = bt.shape
        logits = model(bt.view(B * Kk, L), bs.view(B * Kk, L)).view(B, Kk).masked_fill(~bv, -1e4)
        pos = by >= 0
        correct += (logits[pos].argmax(-1) == by[pos]).sum().item(); total += pos.sum().item()
    return correct / max(total, 1)


def listed_misspellings():
    path = RES / "misspellings_en.txt"
    if not path.exists():
        return []
    return [tuple(line.split("\t")) for line in path.read_text().splitlines() if "\t" in line]


def synthetic_dataset(words, counts, cands, n, seed, tag, real_repeats=0):
    path = CACHE / f"synth_{tag}_{n}_{seed}_r{real_repeats}.pt"
    if path.exists():
        return torch.load(path)
    rng = random.Random(seed)
    targets = sample_words(words, counts, n, rng)
    pairs = [(make_typo(w, rng), w) for w in targets]
    if real_repeats:
        real = listed_misspellings()
        pairs += real * real_repeats
        rng.shuffle(pairs)
        log(f"added {len(real)} curated misspellings x{real_repeats}")
    log(f"built {len(pairs)} synthetic pairs, generating candidate lists…")
    data = build_lists(pairs, cands, rng)
    CACHE.mkdir(exist_ok=True)
    torch.save(data, path)
    return data


def realistic_dataset(words, counts, cands, n, seed, tag, tables=True):
    """Typos from the fitted error model for words drawn by frequency; with `tables`, every typo
    people really typed (keystroke study) and wrote (Birkbeck) is added as well."""
    path = CACHE / f"real_{tag}_{n}_{seed}.pt"
    if path.exists():
        return torch.load(path)
    from noise import Noise
    noise, rng = Noise(), random.Random(seed)
    known = set(words)
    pool = [i for i, (w, c) in enumerate(zip(words, counts)) if c >= 300.0 and len(w) >= 2]
    targets = [words[i] for i in rng.choices(pool, weights=[counts[i] ** 0.5 for i in pool], k=n)]
    pairs = [(t, w) for w in targets if (t := noise.sample(w, rng))]
    if tables:
        real = [(t, w) for w, (typos, seen) in noise.seen.items() if w in known
                for t, c in zip(typos, seen) for _ in range(min(6, c.bit_length()))]
        missp = [(t, w) for w, typos in noise.missp.items() if w in known for t in typos] * 2
        log(f"added {len(real)} real typing slips and {len(missp)} real misspellings")
        pairs += real + missp
        rng.shuffle(pairs)
    log(f"built {len(pairs)} realistic pairs, generating candidate lists…")
    data = build_lists(pairs, cands, rng)
    CACHE.mkdir(exist_ok=True)
    torch.save(data, path)
    return data


def personal_pairs(db_path):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT typo, fix, source, COUNT(*) FROM fixes WHERE source IN ('manual','undo') GROUP BY typo, fix, source").fetchall()
    con.close()
    manual, undo = [], []
    for typo, fix, source, n in rows:
        typo, fix = typo.lower(), fix.lower()
        if not typo.replace("'", "").isalpha() or not fix.replace("'", "").isalpha():
            continue
        (manual if source == "manual" else undo).append(((typo, fix), n))
    return manual, undo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--personal", action="store_true")
    ap.add_argument("--db")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--realistic", action="store_true", help="with --base: fitted error model + real typo tables")
    ap.add_argument("--d", type=int, default=CONFIG["d"])
    ap.add_argument("--layers", type=int, default=CONFIG["layers"])
    ap.add_argument("--ff", type=int, default=CONFIG["ff"])
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    args = ap.parse_args()
    if not (args.base or args.personal):
        ap.error("pass --base or --personal")

    torch.manual_seed(args.seed)
    dev = device()
    if dev.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mem_fraction)   # a bug must fail, not starve the Mac
    words, counts = load_lexicon(RES / "words_en.txt")
    cands = Candidates(words, counts, cache_path=str(CACHE / "symspell_index.pkl"))
    log(f"lexicon {len(words)} words, device {dev}")

    if args.base and args.realistic:
        if not args.out:
            ap.error("--realistic needs --out (the bundled ranker is replaced only on purpose)")
        test = realistic_dataset(words, counts, cands, 20_000, args.seed + 1000, "test", tables=False)
        old = synthetic_dataset(words, counts, cands, 10_000, args.seed + 1000, "test")
        bundled = (RES / "ranker.json", RES / "ranker.bin")
        if bundled[0].exists():
            ref = import_exported(PairRanker(), str(bundled[0]), str(bundled[1])).to(dev)
            log(f"bundled ranker: realistic test top1 {evaluate(ref, test, dev):.3f}, random-slip test top1 {evaluate(ref, old, dev):.3f}")
        train = realistic_dataset(words, counts, cands, args.n, args.seed, "train")
        slips = synthetic_dataset(words, counts, cands, 200_000, args.seed, "train", real_repeats=5)
        train = tuple(torch.cat([a, b.to(a.dtype)]) for a, b in zip(train, slips))   # keep random slips covered
        cfg = dict(CONFIG, version=2, d=args.d, layers=args.layers, ff=args.ff)
        model = PairRanker(cfg).to(dev)
        log(f"params {sum(p.numel() for p in model.parameters())}, {train[0].shape[0]} lists")
        run_epochs(model, train, args.epochs, args.lr, args.batch, dev)
        log(f"realistic test top1 {evaluate(model, test, dev):.3f}, random-slip test top1 {evaluate(model, old, dev):.3f}")
        n = export(model, args.out)
        torch.save(model.state_dict(), Path(args.out) / "base.pt")
        log(f"exported {n} floats to {args.out}/ranker.bin")
    elif args.base:
        train = synthetic_dataset(words, counts, cands, args.n, args.seed, "train", real_repeats=5)
        test = synthetic_dataset(words, counts, cands, 10_000, args.seed + 1000, "test")
        model = PairRanker().to(dev)
        log(f"params {sum(p.numel() for p in model.parameters())}")
        log(f"untrained test top1 {evaluate(model, test, dev):.3f}")
        run_epochs(model, train, args.epochs, args.lr, args.batch, dev)
        log(f"test top1 (neural only, among ≤{K} candidates) {evaluate(model, test, dev):.3f}")
        out = args.out or str(RES)
        n = export(model, out)
        torch.save(model.state_dict(), CACHE / "base.pt")
        log(f"exported {n} floats to {out}/ranker.bin")

    if args.personal:
        if not args.db or not args.out:
            ap.error("--personal needs --db and --out")
        manual, undo = personal_pairs(args.db)
        log(f"personal data: {sum(n for _, n in manual)} manual fixes ({len(manual)} distinct), {sum(n for _, n in undo)} undos")
        if not manual and not undo:
            log("nothing to learn from yet"); return 0
        model = PairRanker()
        base_pt = CACHE / "base.pt"
        bundled = (RES / "ranker.json", RES / "ranker.bin")
        if base_pt.exists():
            model.load_state_dict(torch.load(base_pt, map_location="cpu"))
            log(f"starting from {base_pt}")
        elif bundled[0].exists() and bundled[1].exists():
            import_exported(model, str(bundled[0]), str(bundled[1]))
            log(f"starting from bundled {bundled[0]}")
        else:
            log("error: no base weights (run train.py --base first); refusing to export a random model")
            return 2
        model = model.to(dev)
        rng = random.Random(args.seed)
        # Weight by how often you made each fix, capped so one habit doesn't dominate.
        pos_pairs = [p for p, n in manual for _ in range(min(n, 5))]
        neg_pairs = [p for p, n in undo for _ in range(min(n, 5))]
        reps = max(1, 400 // max(len(pos_pairs) + len(neg_pairs), 1))
        data_pos = build_lists(pos_pairs * reps, cands, rng) if pos_pairs else None
        data_neg = build_lists(neg_pairs * reps, cands, rng, negatives_only=True) if neg_pairs else None
        parts = [d for d in (data_pos, data_neg) if d is not None and d[0].shape[0] > 0]
        if not parts:
            log("no usable examples (fixes need dictionary candidates)"); return 0
        data = tuple(torch.cat([p[i] for p in parts]) for i in range(4))
        replay = synthetic_dataset(words, counts, cands, 20_000, 7, "replay")
        before = evaluate(model, data, dev)
        run_epochs(model, data, epochs=6, lr=3e-4, batch=64, dev=dev, replay=replay, replay_ratio=4)
        after = evaluate(model, data, dev)
        log(f"personal top1 before {before:.3f} after {after:.3f}; replay top1 {evaluate(model, replay, dev):.3f}")
        n = export(model, args.out)
        log(f"exported {n} floats to {args.out}/ranker.bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
