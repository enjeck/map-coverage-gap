"""My hand-built gap estimates (v2b). Every number comes from the challenge data only.

Each gap is max(0, 1 - Overture amount / expected reference amount). The expected reference comes
from a stand-in. Roads use route km, buildings use the USFS building count, and places use
population and area. The score is the expected average over the gaps that are likely to be defined.
Writes out/v2b.csv in Zindi's 17-column layout.
"""
import itertools, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
P = dict(
    tau=2.0,            # km of TIGER highway expected for each km of highway route
    w_fm=1.0,           # weight of Texas Farm to Market routes in the route length
    route_min=0.05,     # below this many km of route, the road gap is undefined
    gamma=1.027,        # reference buildings expected for each USFS building
    fire_rate=1.0, ems_rate=0.5, school_rate=10.0,     # facilities per 10,000 people
    fire_area=0.004, ems_area=0.002, school_area=0.0,  # extra facilities per km2
    kappa=20.0,         # businesses per 1,000 people
)
A = pd.read_parquet(ROOT/"features/all-tracts.parquet")
A = A[A.scored].copy()
tau = P["tau"]
clip = lambda x: np.clip(x, 0, 1)

# Roads
route = A.ov_km_route_hwy.values + P["w_fm"] * A.ov_km_route_txfm.values
road_def = route >= P["route_min"]
g_road = np.where(road_def, clip(1 - A.ov_km_named.values / np.maximum(tau * route, 1e-9)), 0.0)

# Buildings
usfs = A.usfs_BuildingCount_sum.fillna(0).values
bldg_def = usfs >= 1
g_bldg = np.where(bldg_def, clip(1 - A.ov_bldg.values / np.maximum(P["gamma"] * usfs, 1e-9)), 0.0)

# Facilities. The reference count is Poisson with mean lam, and the gap is averaged over every count.
pop = A.pop_total.fillna(0).values; area = A.tract_area_km2.values
def fac(o, lam):
    lam = np.maximum(lam, 1e-6); p_def = 1 - np.exp(-lam)
    k = np.arange(1, 31)[:, None]
    from math import lgamma
    lp = k * np.log(lam)[None, :] - lam[None, :] - np.array([lgamma(x + 1) for x in range(1, 31)])[:, None]
    pk = np.exp(lp); pk = pk / pk.sum(0, keepdims=True)
    g = np.clip(1 - o[None, :] / k, 0, 1)
    return p_def, (pk * g).sum(0)
types = []
for t, col in [("fire", "ov_fire"), ("ems", "ov_ems"), ("school", "ov_school")]:  # order matters below
    lam = P[f"{t}_rate"] * pop / 1e4 + P[f"{t}_area"] * area
    types.append(fac(A[col].values.astype(float), lam))

# Businesses
exp_est = P["kappa"] * pop / 1e3
est_def = exp_est >= 0.5
g_est = np.where(est_def, clip(1 - A.ov_poi_open.values / np.maximum(exp_est, 1e-9)), 0.0)

# The score averages over which facility types exist, weighted by how likely each one is.
n = len(A); E = np.zeros(n); E_poi = np.zeros(n); P_poi = np.zeros(n)
for combo in itertools.product([0, 1], repeat=3):
    pr = np.ones(n); gs = []
    for on, (pd_, g) in zip(combo, types):
        pr *= pd_ if on else (1 - pd_)
        if on: gs.append(g)
    fac_g = np.mean(gs, axis=0) if gs else None
    if fac_g is not None:
        poi = np.where(est_def, (fac_g + g_est) / 2, fac_g); poi_def = np.ones(n, bool)
    else:
        poi = g_est; poi_def = est_def
    comps = [(g_road, road_def), (g_bldg, bldg_def), (poi, poi_def)]
    num = sum(np.where(d, g, 0) for g, d in comps); den = sum(d.astype(float) for g, d in comps)
    E += pr * np.where(den > 0, num / np.maximum(den, 1), 0)
    E_poi += pr * np.where(poi_def, poi, 0); P_poi += pr * poi_def
g_poi = np.where(P_poi > 0, E_poi / np.maximum(P_poi, 1e-9), 0)

score = E
TF = lambda b: np.where(b, "TRUE", "FALSE")
(pf, gf), (pe, ge), (ps, gs_) = types
out = pd.DataFrame({
    "GEOID": A.GEOID.values, "coverage_gap_score": clip(score).round(6), "region": A.region.values,
    "transport_gap": g_road.round(6), "transport_defined": TF(road_def),
    "building_gap": g_bldg.round(6), "building_defined": TF(bldg_def),
    "poi_gap": g_poi.round(6), "poi_defined": TF(P_poi >= 0.5),
    "poi_gap_fire": gf.round(6), "poi_defined_fire": TF(pf >= 0.5),
    "poi_gap_ems": ge.round(6), "poi_defined_ems": TF(pe >= 0.5),
    "poi_gap_schools": gs_.round(6), "poi_defined_schools": TF(ps >= 0.5),
    "poi_gap_cbp": g_est.round(6), "poi_defined_cbp": TF(est_def)})
OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
out.to_csv(OUT/"v2b.csv", index=False)
print("v2b", len(out), "rows, mean", round(out.coverage_gap_score.mean(), 4))
