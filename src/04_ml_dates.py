"""Count the Overture buildings in each tract that came from Microsoft's model, and how many of them are old.

Old means dated 2015 to 2019, or dated 2000, which Overture uses when the date is unknown.
Run python src/04_ml_dates.py <region> [first_row_group last_row_group]
The row-group range lets a large region (Texas) run in two parts.
After all regions, run python src/04_ml_dates.py merge to write features/ml-dates.parquet.
"""
import sys, glob, pathlib, numpy as np, pandas as pd, pyarrow.parquet as pq, shapely
ROOT = pathlib.Path(__file__).resolve().parent.parent
if sys.argv[1] == "merge":
    d = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ROOT/"features/ml-dates-*-*.parquet"))]).groupby("GEOID", as_index=False).sum()
    d.to_parquet(ROOT/"features/ml-dates.parquet", index=False); print(d.shape, d[["ml", "old"]].sum().to_dict()); sys.exit()
R = sys.argv[1]
f = pq.ParquetFile(ROOT/f"data/reference/{R}/{R}-overture-buildings.parquet")
a = int(sys.argv[2]) if len(sys.argv) > 2 else 0; b = int(sys.argv[3]) if len(sys.argv) > 3 else f.metadata.num_row_groups
t = pq.read_table(ROOT/f"data/strata/{R}/{R}-census-tracts.parquet", columns=["GEOID", "geometry"]).to_pydict()
gid = np.array(t["GEOID"]); tree = shapely.STRtree(shapely.from_wkb(t["geometry"]))
ml = np.zeros(len(gid)); old = np.zeros(len(gid))
for rg in range(a, min(b, f.metadata.num_row_groups)):
    tb = f.read_row_group(rg, columns=["bbox", "sources"]).to_pydict()
    ds = np.array([(s[0]["dataset"] if s else "") for s in tb["sources"]]); m = ds == "Microsoft ML Buildings"
    if not m.any(): continue
    yr = np.array([(s[0]["update_time"] or "")[:4] for s, k in zip(tb["sources"], m) if k])
    bb = [x for x, k in zip(tb["bbox"], m) if k]
    x = np.array([(q["xmin"] + q["xmax"]) / 2 for q in bb]); y = np.array([(q["ymin"] + q["ymax"]) / 2 for q in bb])
    pi, ti = tree.query(shapely.points(x, y), predicate="within")
    np.add.at(ml, ti, 1); np.add.at(old, ti, np.isin(yr[pi], ["2000", "2017", "2018", "2019", "2016", "2015"]).astype(float))
out = pd.DataFrame({"GEOID": gid, "ml": ml, "old": old}); out = out[out.ml > 0]
out.to_parquet(ROOT/f"features/ml-dates-{R}-{a}.parquet", index=False); print(R, a, b, int(ml.sum()), int(old.sum()))
