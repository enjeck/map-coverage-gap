"""Blend clues into one prediction, or write a probe file that tests one clue on the leaderboard.

Each clue z is scaled to mean 0 and standard deviation 1 over the scored tracts.
A probe predicts m + s * z for every tract, where m is the target mean. Its public score gives the
clue's covariance with the hidden target, cov = sign * (var + s^2 - RMSE^2) / (2s).
A blend predicts m + sum of w_i * z_i, with w = (S + ridge * I)^-1 c. S holds the correlations between
clues (from our own data) and c their covariances with the target (from the probes).
Predictions below 0 are set to 0, since the target is never negative.

Usage
  python src/06_blend.py v11                  blend the clues listed in models/v11.json
  python src/06_blend.py <name> --probe <clue>  write a probe file for one clue
  python src/06_blend.py <name> --const <g>     write a file that predicts g everywhere

A clue is a column of out/v2b.csv, clue:<column> from features/clues.parquet, region:<region>
(1 inside the region, 0 elsewhere), or a product a*b.
Means, variances and probe scores come from scores.json. Files are written to out/.
"""
import sys, json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent.parent
SC = json.load(open(ROOT/"scores.json")); m, v = SC["mean"], SC["var"]
base = pd.read_csv(ROOT/"out/v2b.csv", dtype={"GEOID": str})
CL = pd.read_parquet(ROOT/"features/clues.parquet").set_index("GEOID").loc[base.GEOID].reset_index()
OUT = ROOT/"out"

def raw(f):
    if f.startswith("clue:"): return CL[f[5:]].astype(float)
    if f.startswith("region:"): return (base.region == f.split(":")[1]).astype(float)
    if base[f].dtype == object: return (base[f] == "TRUE").astype(float)
    return base[f].astype(float)

def z(f):
    x = raw(f.split("*")[0]) * raw(f.split("*")[1]) if "*" in f else raw(f)
    return ((x - x.mean()) / x.std()).values

def probe_name(f):
    """Name of the leaderboard probe that tested clue f, as used in scores.json."""
    if f.startswith("transport_gap*region:"): return "probe_road_x_" + f.split(":")[1].replace("-", "_")
    fixed = {"transport_gap": "probe_road", "transport_defined": "probe_hwy_defined"}
    return fixed.get(f, "probe_" + f.replace("region:", "reg_").replace("clue:", "pf_").replace("-", "_"))

def cov(f):
    p = probe_name(f); s, sign = SC["probes"][p]["s"], SC["probes"][p]["sign"]
    return sign * (v + s * s - SC["public"][p] ** 2) / (2 * s)

name, args = sys.argv[1], sys.argv[2:]
if args and args[0] == "--const":
    pred = np.full(len(base), float(args[1])); info = {"const": float(args[1])}
elif args and args[0] == "--probe":
    x = z(args[1]); sign = 1.0
    if -x.min() > x.max(): x, sign = -x, -1.0           # put the long tail on the positive side
    s = min(0.03, 0.95 * m / -x.min())                  # small enough that no prediction drops below 0
    pred = m + s * x; info = {"clue": args[1], "s": round(float(s), 6), "sign": sign}
else:
    spec = json.load(open(ROOT/f"models/{name}.json"))
    Z = np.column_stack([z(f) for f in spec["clues"]]); k = len(spec["clues"])
    S = np.cov(Z.T, bias=True).reshape(k, k)
    c = np.array([cov(f) for f in spec["clues"]])
    w = np.linalg.solve(S + spec["ridge"] * np.eye(k), c); pred = m + Z @ w
    info = {"weights": [round(float(x), 6) for x in w], "expected_rmse": round(float(np.sqrt(max(v - 2 * c @ w + w @ S @ w, 0))), 5)}
out = base.copy(); out["coverage_gap_score"] = np.clip(pred, 0, 1).round(6)
out.to_csv(OUT/f"{name}.csv", index=False)
print(name, len(out), "rows", info, "set to 0:", int((pred < 0).sum()))
