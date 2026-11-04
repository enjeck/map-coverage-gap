"""Print every number in the bias writeup about missing buildings in northeast Oklahoma.

Needs features/all-tracts.parquet (from 02_build_table.py) and the CDC mobile home file in data/external.
The ratio is Overture buildings divided by the USFS building count. A shortfall is a ratio below 0.9.
"""
import json, pathlib, numpy as np, pandas as pd, pyarrow.parquet as pq, shapely, duckdb
from scipy.stats import norm, spearmanr
ROOT = pathlib.Path(__file__).resolve().parent.parent; D = ROOT/"data"
pos = lambda s: s.where(s > 0)
say = lambda label, value: print(f"{label:<62} {value}")
r = lambda x, n=2: round(float(x), n)

A = pd.read_parquet(ROOT/"features/all-tracts.parquet"); A = A[A.scored].reset_index(drop=True)
M = pd.read_csv(D/"external/svi2022_tract_mobile_homes.csv", dtype={"GEOID": str})
A = A.merge(M, on="GEOID", how="left")
A["ratio"] = A.ov_bldg / pos(A.usfs_BuildingCount_sum)
A["short"] = A.ratio < 0.9
A["rural"] = A.ur_class == "Rural"
A["msml"] = A.ov_bldg_msml / pos(A.ov_bldg)
A["mh"] = (A.e_mobile / pos(A.housing_units)).clip(upper=1)
A["water"] = 1 - A.ALAND / (A.tract_area_km2 * 1e6)

print("All scored tracts")
both = A[(A.ov_bldg > 0) & (A.usfs_BuildingCount_sum > 0)]
say("tracts", len(A))
say("correlation of log counts, Overture vs USFS", round(np.corrcoef(np.log(both.ov_bldg), np.log(both.usfs_BuildingCount_sum))[0, 1], 2))
say("median ratio", round(both.ratio.median(), 2))

# Getis-Ord Gi* on the log ratio in eastern Oklahoma. Neighbours share a border or a corner (queen),
# weights are 1 and include the tract itself, and z scores are analytic.
R = "eastern-ok"; ok = A[A.region == R].reset_index(drop=True)
t = pq.read_table(D/f"strata/{R}/{R}-census-tracts.parquet", columns=["GEOID", "geometry"]).to_pydict()
geom = dict(zip(t["GEOID"], shapely.from_wkb(t["geometry"])))
g = np.array([geom[x] for x in ok.GEOID]); a, b = shapely.STRtree(g).query(g, predicate="intersects")
nb = [[] for _ in g]
for i, j in zip(a, b):
    if i != j: nb[i].append(j)
x = np.log(ok.ratio.values.astype(float)); valid = ~np.isnan(x); n = valid.sum(); xm, s = x[valid].mean(), x[valid].std()
zs = np.full(len(ok), np.nan)   # z scores are rounded to 2 decimals, as in my original run
for i in np.where(valid)[0]:
    js = [j for j in nb[i] if valid[j]] + [i]; w = len(js)
    zs[i] = round((x[js].sum() - xm * w) / (s * np.sqrt((n * w - w * w) / (n - 1))), 2)
cold = np.nan_to_num(zs, nan=0) < -1.96

# The cluster is the largest connected group of cold-spot tracts.
seen = np.zeros(len(ok), bool); groups = []
for i in np.where(cold)[0]:
    if seen[i]: continue
    stack, grp = [i], []; seen[i] = True
    while stack:
        k = stack.pop(); grp.append(k)
        for j in nb[k]:
            if cold[j] and not seen[j]: seen[j] = True; stack.append(j)
    groups.append(grp)
C = sorted(groups, key=len, reverse=True)[0]
ok["cluster"] = False; ok.loc[C, "cluster"] = True
c, rest = ok[ok.cluster], ok[~ok.cluster]

# Benjamini-Hochberg on two-sided p values across all tested tracts.
p = 2 * norm.sf(np.abs(zs[valid])); order = np.argsort(p); q = np.empty_like(p)
q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
qs = np.full(len(ok), np.nan); qs[valid] = q

print("\nThe cluster")
say("tracts", len(c))
say("tracts still significant after Benjamini-Hochberg", int((qs[C] < 0.05).sum()))
say("tracts by county (FIPS)", c.COUNTYFP.value_counts().to_dict())
say("tracts in the Cherokee tribal statistical area", int((c.aiannh_name == "Cherokee OTSA").sum()))
say("Overture buildings", int(c.ov_bldg.sum()))
say("USFS buildings", round(c.usfs_BuildingCount_sum.sum()))
say("Overture buildings for each USFS building", round(c.ov_bldg.sum() / c.usfs_BuildingCount_sum.sum(), 3))
say("lowest tract and its ratio", (c.loc[c.ratio.idxmin(), "GEOID"], r(c.ratio.min())))
say("Overture buildings from Microsoft / from OpenStreetMap", (int(c.ov_bldg_msml.sum()), int(c.ov_bldg_osm.sum())))
say("share of Overture buildings from Microsoft", round(c.ov_bldg_msml.sum() / c.ov_bldg.sum(), 2))
say("people / homes", (int(c.pop_total.sum()), int(c.housing_units.sum())))
say("rural tracts and their ratio", (int(c.rural.sum()), r(c[c.rural].ov_bldg.sum() / c[c.rural].usfs_BuildingCount_sum.sum())))
say("share of homes that are mobile homes", round(c.e_mobile.sum() / c.housing_units.sum(), 2))
say("median social vulnerability, cluster", round(c.svi_overall.median(), 2))
say("median social vulnerability, other rural tracts", round(rest[rest.rural].svi_overall.median(), 2))

print("\nComparison group: other rural eastern Oklahoma tracts with at least 90% Microsoft buildings")
cmp_ = rest[rest.rural & (rest.msml >= 0.9)]
say("tracts", len(cmp_))
say("Overture buildings for each USFS building", round(cmp_.ov_bldg.sum() / cmp_.usfs_BuildingCount_sum.sum(), 2))
say("share of Overture buildings from Microsoft", round(cmp_.ov_bldg_msml.sum() / cmp_.ov_bldg.sum(), 2))

# Dates on Overture's Microsoft buildings. 2000 is Overture's placeholder for an unknown date.
df = duckdb.sql(f"""select (bbox.xmin+bbox.xmax)/2 x, (bbox.ymin+bbox.ymax)/2 y, substr(sources[1].update_time,1,4) yr
    from '{D}/reference/{R}/{R}-overture-buildings.parquet' where sources[1].dataset='Microsoft ML Buildings'""").df()
pi, ti = shapely.STRtree(g).query(shapely.points(df.x.values, df.y.values), predicate="within")
df["GEOID"] = None; df.loc[df.index[pi], "GEOID"] = ok.GEOID.values[ti]
grp = pd.Series("other", index=ok.GEOID.values); grp[c.GEOID.values] = "cluster"; grp[cmp_.GEOID.values] = "comparison"
df["grp"] = df.GEOID.map(grp)
print("\nMicrosoft building dates")
for name in ["cluster", "comparison"]:
    yr = df[df.grp == name].yr
    say(f"{name}, share dated 2020 or later", r((yr >= "2020").mean(), 3))
    say(f"{name}, share dated 2000 (unknown date)", r((yr == "2000").mean(), 3))
df["old"] = df.yr.isin(["2000", "2017", "2018", "2019"])
per = df.dropna(subset=["GEOID"]).groupby("GEOID").agg(ml=("old", "size"), old=("old", "mean"))
j = ok.set_index("GEOID").join(per, how="left"); j = j[j.ml >= 50]
say("Spearman, old share vs log ratio (50+ Microsoft buildings)", r(spearmanr(j.old, np.log(j.ratio), nan_policy="omit")[0]))
say("same, rural tracts only", r(spearmanr(j[j.rural].old, np.log(j[j.rural].ratio), nan_policy="omit")[0]))

print("\nTribal land, eastern Oklahoma")
trib = ok.tribal_pct > 0
say("share of tracts touching tribal land", round(trib.mean(), 2))
say("shortfall share, tribal / other tracts", (r(ok[trib].short.mean()), r(ok[~trib].short.mean())))

print("\nOttawa and Craig counties (tornado example)")
oc = c[c.COUNTYFP.isin(["115", "035"])]
say("cluster tracts / Overture buildings / USFS buildings", (len(oc), int(oc.ov_bldg.sum()), round(oc.usfs_BuildingCount_sum.sum())))

print("\nGrand Lake check")
dry = c[c.water <= 0.05]
say("cluster tracts more than 5% water", int((c.water > 0.05).sum()))
say("other tracts / their ratio", (len(dry), r(dry.ov_bldg.sum() / dry.usfs_BuildingCount_sum.sum(), 3)))
say("USFS minus Overture in those tracts", round(dry.usfs_BuildingCount_sum.sum() - dry.ov_bldg.sum()))

print("\nMobile homes, rural tracts in all five regions")
Dm = A[A.ratio.gt(0) & A.mh.notna() & A.svi_overall.notna() & A.pop_total.gt(0)]
Dm = Dm[Dm.rural].copy()
Dm["bin"] = pd.cut(Dm.mh, [0, .02, .05, .10, .20, .35, 1.01], labels=["0-2%", "2-5%", "5-10%", "10-20%", "20-35%", "35%+"], right=False)
for k, v in Dm.groupby("bin", observed=True): say(f"mobile homes {k}, tracts / shortfall share", (len(v), r(v.short.mean())))

print("\nSocial vulnerability fifths, all scored tracts")
Dq = A.dropna(subset=["ratio", "svi_overall"]).copy(); Dq["q"] = pd.qcut(Dq.svi_overall, 5, labels=False)
for label, v in [("rural", Dq[Dq.rural]), ("urban", Dq[~Dq.rural])]:
    say(f"{label}, shortfall share from least to most vulnerable", v.groupby("q").short.mean().round(2).tolist())

print("\nThe 59 tracts"); print(" ".join(sorted(c.GEOID)))
