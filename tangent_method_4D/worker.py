"""Worker: one manifest task = one chunk of raw t-statistics -> one npz (atomic,
idempotent). Rebuilds the tangent feature banks deterministically at startup."""
import argparse, csv, os, time
import numpy as np
import gof4d as G
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=C.MANIFEST)
ap.add_argument("--task-id", type=int,
                default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
args = ap.parse_args()
assert args.task_id >= 0

with open(args.manifest) as f:
    row = list(csv.DictReader(f))[args.task_id]
key = row["key"]
fname = os.path.join(C.RAW_DIR, key.replace("|", "_") + ".npz")
os.makedirs(C.RAW_DIR, exist_ok=True)
ledger = os.path.join(C.OUT_DIR, "compacted_keys.txt")
done = set()
if os.path.exists(ledger):
    done = set(open(ledger).read().split())
if os.path.exists(fname) or key in done:
    print("done already, skipping:", key); raise SystemExit(0)

test_type, source = row["test_type"], row["source"]
N, K, nt = int(row["N_fit"]), int(row["K"]), int(row["N_test"])
eps, n_items = float(row["eps"]), int(row["n_items"])

art = np.load(C.ARTIFACTS, allow_pickle=True)
tan = G.rehydrate_tangent(art["tans"].item()[(N, K)], n_test=nt)   # regenerates R0 + banks

rng = G.rng_for(key)
t0 = time.time()
tvals = np.empty(n_items)
for i in range(n_items):
    if source in ("calib", "null"):
        th_b = (G.sample_theta_tan(tan, rng)
                if test_type == "comp" and C.THETA_SAMPLED_CALIB else tan["theta"])
        X = G.gmm_rvs(th_b, K, nt, rng)
    elif source == "truth":
        X = G.sample_data(nt, rng, N_excl=N)
    elif source == "power":
        X = G.sample_data(nt, rng, N_excl=N, eps=eps)
    else:
        raise ValueError(source)
    tvals[i] = G.t_point_tan(X, tan) if test_type == "point" else G.t_comp_tan(X, tan)

tmp = fname + ".tmp.npz"
np.savez(tmp, t=tvals, key=key, test_type=test_type, source=source,
         N_fit=N, K=K, N_test=nt, eps=eps, chunk=int(row["chunk"]),
         wall_s=time.time() - t0, base_seed=C.BASE_SEED)
os.replace(tmp, fname)
print(f"{key}: {n_items} items in {time.time()-t0:.1f}s -> {fname}")
