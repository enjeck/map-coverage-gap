"""Build the extra clues used by the blend. Writes features/clues.parquet in the row order of out/v2b.csv.

Run after 02_build_table.py, 03_hand_gaps.py and 04_ml_dates.py.
"""
import pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent.parent
b = pd.read_csv(ROOT/"out/v2b.csv", dtype={"GEOID": str})
A = pd.read_parquet(ROOT/"features/all-tracts.parquet").set_index("GEOID").loc[b.GEOID].reset_index()
M = pd.read_parquet(ROOT/"features/ml-dates.parquet").set_index("GEOID").reindex(b.GEOID).fillna(0).reset_index()
pos = lambda s: s.where(s > 0)
F = pd.DataFrame({"GEOID": b.GEOID})
route = A.ov_km_route_hwy + A.ov_km_route_txfm; named = A.ov_km_named; d = route >= 0.05; reg = A.region.values
F["osm_share"] = (A.ov_bldg_osm / pos(A.ov_bldg)).fillna(0)
F["no_fire"] = (A.ov_fire == 0).astype(float)
F["no_school"] = (A.ov_school == 0).astype(float)
F["no_ems"] = (A.ov_ems == 0).astype(float)
F["route_minor_share"] = ((route - A.ov_km_route_hwy_named) / pos(route)).clip(0, 1).fillna(0)
F["log_places_pc"] = np.log1p(A.ov_poi_open / pos(A.pop_total) * 1000).fillna(0)
F["log_density"] = np.log10(pos(A.pop_total) / A.tract_area_km2).fillna(0)
F["ml_old_frac_all"] = (M.old / pos(A.ov_bldg)).fillna(0).clip(0, 1)
F["msml_share"] = (A.ov_bldg_msml / pos(A.ov_bldg)).fillna(0)
F["log_route_dens"] = np.log1p(route / A.tract_area_km2)
F["log_named_ratio"] = np.log((named + 0.2) / (route + 0.2))
F["has_interstate"] = (A.ov_km_route_interstate > 0.05).astype(float)
for t in [1.0, 1.5, 3.0, 4.0, 6.0]:
    F[f"road_gap_t{str(t).replace('.', '_')}"] = np.where(d, np.clip(1 - named / np.maximum(t * route, 1e-9), 0, 1), 0)
F["road_gap_t3_x_ok"] = F.road_gap_t3_0 * (reg == "eastern-ok")
F["road_gap_t3_x_tx"] = F.road_gap_t3_0 * (reg == "south-central-tx")
F.to_parquet(ROOT/"features/clues.parquet", index=False); print("clues", F.shape)
