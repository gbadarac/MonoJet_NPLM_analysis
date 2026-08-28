"""Merge raw chunk files into analysis tables; optionally COMPACT (fold raw npz into
the single tstats table + a key ledger, then delete them - resumability preserved:
workers treat 'key in ledger' as done).

  python merge.py              # merge only (raw files kept)
  python merge.py --compact    # merge + delete absorbed raw files

Produces (in <run>/results/): tstats.parquet (all raw t values, with their task key),
pvalues.parquet (one row per observed t with its p vs the matching calib bank),
summary.csv (per-working-point quantiles), compacted_keys.txt (ledger, --compact only).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import config as C  # GOF*_PARAMS-driven

ap = argparse.ArgumentParser()
ap.add_argument("--compact", action="store_true")
args = ap.parse_args()

os.makedirs(C.OUT_DIR, exist_ok=True)
ledger_path = os.path.join(C.OUT_DIR, "compacted_keys.txt")
ledger = set(open(ledger_path).read().split()) if os.path.exists(ledger_path) else set()

def load_tstats():
    p = os.path.join(C.OUT_DIR, "tstats")
    if os.path.exists(p + ".parquet"):
        return pd.read_parquet(p + ".parquet")
    if os.path.exists(p + ".pkl"):
        return pd.read_pickle(p + ".pkl")
    return None

existing = load_tstats()
if existing is not None and "key" not in existing.columns:
    existing["key"] = ""          # legacy tables (pre-compaction era)
# CRITICAL dedupe rule: the saved table is authoritative ONLY for compacted keys
# (whose raw files are gone). Rows from non-compacted keys are rebuilt from the raw
# files below - otherwise every merge run would re-append them (duplicating banks).
if existing is not None:
    n0 = len(existing)
    existing = existing[existing.key.isin(ledger)] if ledger else existing.iloc[0:0]
    if len(existing) != n0:
        print(f"note: {n0 - len(existing)} non-compacted rows in the saved table will "
              "be rebuilt from raw files (dedupe)")

new_rows, absorbed = [], []
for f in sorted(glob.glob(os.path.join(C.RAW_DIR, "*.npz"))):
    z = np.load(f, allow_pickle=True)
    key = str(z["key"])
    if key in ledger:
        absorbed.append(f)        # leftover already in the table
        continue
    for t in np.atleast_1d(z["t"]):
        new_rows.append(dict(key=key, test_type=str(z["test_type"]),
                             source=str(z["source"]), N_fit=int(z["N_fit"]),
                             K=int(z["K"]), N_test=int(z["N_test"]),
                             eps=float(z["eps"]), t=float(t)))
    absorbed.append(f)

df_new = pd.DataFrame(new_rows)
df = pd.concat([d for d in (existing, df_new) if d is not None and len(d)],
               ignore_index=True) if (existing is not None or len(df_new)) else df_new
print(f"tstats: {0 if existing is None else len(existing)} existing + {len(df_new)} new "
      f"= {len(df)} rows  ({len(absorbed)} raw files scanned)")

def save(d, name):
    path = os.path.join(C.OUT_DIR, name)
    try:
        d.to_parquet(path + ".parquet"); print("wrote", path + ".parquet")
    except Exception as e:
        d.to_pickle(path + ".pkl"); print(f"pyarrow unavailable; wrote {path}.pkl")

save(df, "tstats")

group_cols = ["test_type", "N_fit", "K", "N_test"]
prow, srow = [], []
for keys, g in df.groupby(group_cols):
    bank = g[g.source == "calib"].t.to_numpy()
    if len(bank) == 0:
        continue
    for source, go in g[g.source != "calib"].groupby("source"):
        for eps, ge in go.groupby("eps"):
            p = (np.array([(bank >= t).sum() for t in ge.t]) + 1) / (len(bank) + 1)
            base = dict(zip(group_cols, keys), source=source, eps=eps)
            prow += [dict(base, t=t, p=pp) for t, pp in zip(ge.t, p)]
            q25, q50, q75 = np.percentile(p, [25, 50, 75])
            srow.append(dict(base, n_obs=len(p), n_bank=len(bank),
                             p25=q25, p50=q50, p75=q75, p_floor=1/(len(bank)+1)))
save(pd.DataFrame(prow), "pvalues")
sm = pd.DataFrame(srow).sort_values(group_cols) if srow else pd.DataFrame()
sm.to_csv(os.path.join(C.OUT_DIR, "summary.csv"), index=False)
print("wrote", os.path.join(C.OUT_DIR, "summary.csv"))

if args.compact and absorbed:
    new_keys = sorted({r["key"] for r in new_rows})
    with open(ledger_path, "a") as fh:
        for k in new_keys:
            fh.write(k + "\n")
    for f in absorbed:
        os.remove(f)
    print(f"compacted: {len(absorbed)} raw files removed, "
          f"{len(new_keys)} keys added to ledger "
          f"({len(ledger) + len(new_keys)} total)")
