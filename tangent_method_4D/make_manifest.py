"""Expand the grid into manifest.csv (1 row = 1 worker task = 1 chunk of t values).
Content-keyed: regenerating with a larger grid never invalidates existing outputs."""
import argparse, csv
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--coverage", action="store_true")
ap.add_argument("--power", type=str, default="")
args = ap.parse_args()
eps_list = [float(x) for x in args.power.split(",") if x] if args.power else []


def chunks(total, size):
    out, i = [], 0
    while total > 0:
        n = min(size, total); out.append((i, n)); total -= n; i += 1
    return out

rows = []
def add(test_type, source, N, K, nt, n_total, eps=0.0):
    for ci, n in chunks(n_total, C.chunk_size(test_type, nt)):
        key = f"{test_type}|{source}|N{N}|K{K}|T{nt}|eps{eps:g}|c{ci}"
        rows.append(dict(key=key, test_type=test_type, source=source,
                         N_fit=N, K=K, N_test=nt, eps=eps, chunk=ci, n_items=n))

for N in C.N_FIT_LIST:
    for nt in C.NTEST_GRID:
        for K in C.K_LIST:                       # point null: full K sweep
            add("point", "calib", N, K, nt, C.POINT_B_CALIB)
            add("point", "truth", N, K, nt, C.POINT_REPEATS)
        for K in C.COMP_K_LIST:                  # composite: ALL requested K
            add("comp", "calib", N, K, nt, C.COMP_B_CALIB)
            add("comp", "truth", N, K, nt, C.COMP_REPEATS)
            if args.coverage:
                add("comp", "null", N, K, nt, C.COMP_REPEATS)
            for eps in eps_list:
                add("comp", "power", N, K, nt, C.COMP_REPEATS, eps=eps)

with open(C.MANIFEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"{C.MANIFEST}: {len(rows)} tasks "
      f"(point {sum(r['test_type']=='point' for r in rows)}, "
      f"comp {sum(r['test_type']=='comp' for r in rows)})")
