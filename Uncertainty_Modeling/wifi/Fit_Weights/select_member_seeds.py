#!/usr/bin/env python
"""
Member-seed selection for the wifi ensemble (kernels or NF). Prints the chosen seeds
as a comma-separated list to STDOUT (nothing else); diagnostics go to STDERR, so a
caller can capture the list with $(...). Each seed s maps to the member subdir of the
ensemble (kernels: seed<s:03d>/, NF: model_<s:03d>/).

Modes:
  uniform     : draw M seeds uniformly at random WITHOUT replacement from the pool
                {0..n_members-1}. What an ensemble literally is -> the natural baseline.
                Needs --n_members. Works for kernels AND NF (no ranking required).  [DEFAULT]
  stratified  : split by the MARGINAL ranking into best=top42 / medium=mid44 / worst=bottom42
                and draw a fixed (worst,medium,best) count per M. Needs --ranking_file. (NF)
  pinned      : take the first M seeds of the ranking (best->worst), deterministic.
                Needs --ranking_file. (NF)

uniform/stratified draws are reproducible: RNG seeded by (rng_seed, M). pinned is
deterministic (no RNG).
"""
import sys, argparse
import numpy as np

# stratified: M -> (n_worst, n_medium, n_best); strata sizes (best,medium,worst)=(42,44,42)
COUNTS = {
    1: (0, 1, 0), 2: (0, 2, 0), 4: (1, 2, 1), 8: (2, 4, 2), 16: (5, 6, 5),
    32: (10, 12, 10), 64: (21, 22, 21), 128: (42, 44, 42),
}
N_BEST, N_MEDIUM, N_WORST = 42, 44, 42


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_ranking(path):
    if path is None:
        eprint("ERROR: this mode needs --ranking_file")
        sys.exit(1)
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            order.append(int(line.split()[0]))
    seen = set()
    return [s for s in order if not (s in seen or seen.add(s))]  # best -> worst


def counts_for(M):
    if M in COUNTS:
        return COUNTS[M]
    k = M // 3
    return (k, M - 2 * k, k)  # (worst, medium, best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uniform", "stratified", "pinned"], default="uniform")
    ap.add_argument("--ranking_file", default=None)
    ap.add_argument("--n_members", type=int, default=None)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng([args.rng_seed, args.M])

    if args.mode == "uniform":
        if args.n_members is None:
            eprint("ERROR: uniform mode needs --n_members")
            sys.exit(1)
        if args.M > args.n_members:
            eprint(f"ERROR: M={args.M} > n_members={args.n_members}")
            sys.exit(1)
        selected = sorted(rng.choice(args.n_members, size=args.M, replace=False).tolist())
        eprint(f"[select] mode=uniform M={args.M} rng_seed={args.rng_seed}  "
               f"(uniform draw from {args.n_members} members)")
        eprint(f"[select]   seeds: {selected}")

    elif args.mode == "pinned":
        order = load_ranking(args.ranking_file)
        if args.M > len(order):
            eprint(f"ERROR: M={args.M} > {len(order)} ranked members")
            sys.exit(1)
        selected = order[:args.M]
        eprint(f"[select] mode=pinned M={args.M}  (top-{args.M} of the ranking, best->worst)")
        eprint(f"[select]   seeds: {selected}")

    else:  # stratified
        order = load_ranking(args.ranking_file)
        if len(order) != N_BEST + N_MEDIUM + N_WORST:
            eprint(f"WARNING: ranking has {len(order)} seeds; strata sizes assume 128.")
        best = order[:N_BEST]
        medium = order[N_BEST:N_BEST + N_MEDIUM]
        worst = order[N_BEST + N_MEDIUM:]
        n_worst, n_medium, n_best = counts_for(args.M)
        if n_worst + n_medium + n_best != args.M:
            eprint(f"ERROR: counts {(n_worst, n_medium, n_best)} sum != M={args.M}")
            sys.exit(1)
        for name, pool, n in [("worst", worst, n_worst), ("medium", medium, n_medium),
                              ("best", best, n_best)]:
            if n > len(pool):
                eprint(f"ERROR: need {n} from '{name}' but pool has {len(pool)}")
                sys.exit(1)
        pw = sorted(rng.choice(worst, size=n_worst, replace=False).tolist()) if n_worst else []
        pm = sorted(rng.choice(medium, size=n_medium, replace=False).tolist()) if n_medium else []
        pb = sorted(rng.choice(best, size=n_best, replace=False).tolist()) if n_best else []
        selected = pw + pm + pb
        eprint(f"[select] mode=stratified M={args.M} rng_seed={args.rng_seed}  "
               f"(worst,medium,best)=({n_worst},{n_medium},{n_best})")
        eprint(f"[select]   worst : {pw}")
        eprint(f"[select]   medium: {pm}")
        eprint(f"[select]   best  : {pb}")

    print(",".join(str(s) for s in selected))


if __name__ == "__main__":
    main()
