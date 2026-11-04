"""Join the Overture counts, the strata tables, ACS housing and the list of scored tracts.

Writes features/all-tracts.parquet with one row per tract in all five regions.
"""
import pathlib, pandas as pd, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; D = ROOT/"data"; F = ROOT/"features"
REG = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx", "eastern-wa"]
parts = []
for r in REG:
    f = pd.read_parquet(F/f"{r}-overture-features.parquet")
    s = pd.read_parquet(D/f"strata/{r}/{r}-strata-tract-table.parquet")
    h = pd.read_parquet(D/f"reference/{r}/{r}-census-acs-housing.parquet", columns=["GEOID", "housing_units"])
    sub = pd.read_csv(D/f"reference/{r}/{r}-sample-submission.csv", dtype={"GEOID": str})[["GEOID"]]
    sub["scored"] = True
    d = f.merge(s, on="GEOID", how="left").merge(h, on="GEOID", how="left").merge(sub, on="GEOID", how="left")
    d["scored"] = d["scored"].eq(True)
    parts.append(d)
A = pd.concat(parts, ignore_index=True)
A.to_parquet(F/"all-tracts.parquet", index=False)
print(A.shape, A.scored.sum())
