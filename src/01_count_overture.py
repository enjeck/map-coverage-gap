"""Count Overture buildings, places and road length in each Census tract of one region.

Run python src/01_count_overture.py <region> [stage ...]
The stages are bldg, poi, road, route and merge, and all of them run by default. Each stage saves
its own file, so a large region can be run one stage at a time.
Writes features/<region>-overture-features.parquet with one row per tract.
"""
import sys, time, pathlib
import numpy as np, pandas as pd, duckdb, pyarrow.parquet as pq, shapely
from pyproj import Transformer

R = sys.argv[1]
STAGES = sys.argv[2:] or ["bldg", "poi", "road", "route", "merge"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data"
OUT = ROOT / "features"; OUT.mkdir(exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{R} {time.time()-t0:6.0f}s]", *a, flush=True)

# Tracts. Overture stores longitude first, so always_xy keeps x as longitude.
tt = pq.read_table(D/f"strata/{R}/{R}-census-tracts.parquet", columns=["GEOID", "geometry"]).to_pydict()
gid = np.array(tt["GEOID"]); tg = shapely.from_wkb(tt["geometry"]); n = len(gid)
tree = shapely.STRtree(tg)
TF = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)   # EPSG:5070 is equal-area, in metres
proj = lambda g: shapely.transform(g, lambda c: np.column_stack(TF.transform(c[:, 0], c[:, 1])))
tg_p = proj(tg); tree_p = shapely.STRtree(tg_p)
con = duckdb.connect()

def assign(x, y):
    """Tract index for each point, or -1 if it is outside every tract."""
    pi, ti = tree.query(shapely.points(x, y), predicate="within")
    out = np.full(len(x), -1); out[pi] = ti
    return out

def add_counts(F, name, idx, mask=None):
    m = idx >= 0 if mask is None else (idx >= 0) & mask
    F[name] = F.get(name, 0) + np.bincount(idx[m], minlength=n)

def save(F, stage):
    F.to_parquet(OUT/f"{R}-part-{stage}.parquet", index=False)

# Buildings go to the tract that contains the centre of their bounding box.
if "bldg" in STAGES:
    F = pd.DataFrame({"GEOID": gid})
    q = con.sql(f"""select (bbox.xmin+bbox.xmax)/2 x, (bbox.ymin+bbox.ymax)/2 y, coalesce(sources[1].dataset,'none') ds
        from '{D}/reference/{R}/{R}-overture-buildings.parquet'""")
    for b in q.to_arrow_reader(500_000):
        df = b.to_pandas(); idx = assign(df.x.values, df.y.values)
        add_counts(F, "ov_bldg", idx)
        add_counts(F, "ov_bldg_msml", idx, (df.ds == "Microsoft ML Buildings").values)
        add_counts(F, "ov_bldg_osm", idx, (df.ds == "OpenStreetMap").values)
    log("buildings done")
    save(F, "bldg")

# Places, with fire stations, EMS stations and schools counted by category.
if "poi" in STAGES:
    F = pd.DataFrame({"GEOID": gid})
    p = con.sql(f"""select (bbox.xmin+bbox.xmax)/2 x, (bbox.ymin+bbox.ymax)/2 y,
        coalesce(categories.primary,'') cat, coalesce(basic_category,'') bc, coalesce(operating_status,'') st
        from '{D}/reference/{R}/{R}-overture-pois.parquet'""").df()
    idx = assign(p.x.values, p.y.values)
    cat, bc = p.cat.str.lower(), p.bc.str.lower()
    fire = (bc == "fire_station") | cat.isin(["fire_department", "fire_station"])
    ems = cat.str.contains("ambulance|emergency_medical|paramedic") | bc.str.contains("ambulance")
    school = cat.isin(["school", "elementary_school", "middle_school", "high_school", "charter_school", "public_school",
                       "private_school", "montessori_school", "kindergarten", "preschool", "religious_school"])
    add_counts(F, "ov_poi_open", idx, (p.st != "permanently_closed").values)
    add_counts(F, "ov_fire", idx, fire.values)
    add_counts(F, "ov_ems", idx, ems.values)
    add_counts(F, "ov_school", idx, school.values)
    log("places done", len(p))
    save(F, "poi")

# Main road km, from the four classes the organisers compare with TIGER. Segments are cut at tract borders.
if "road" in STAGES:
    F = pd.DataFrame({"GEOID": gid})
    MAIN = ["motorway", "trunk", "primary", "secondary"]
    r = con.sql(f"select class, geometry from '{D}/reference/{R}/{R}-overture-roads.parquet' where subtype='road'").df()
    rg = proj(shapely.from_wkb([bytes(b) for b in r.geometry.values])); rc = r["class"].values
    for c in MAIN: F[f"km_{c}"] = 0.0
    for s in range(0, len(rg), 200_000):
        g = rg[s:s+200_000]; cl = rc[s:s+200_000]
        gi, ti = tree_p.query(g, predicate="intersects")
        L = shapely.length(shapely.intersection(g[gi], tg_p[ti])) / 1000
        for c in MAIN:
            m = cl[gi] == c; F[f"km_{c}"] += np.bincount(ti[m], weights=L[m], minlength=n)
    F["ov_km_named"] = F[[f"km_{c}" for c in MAIN]].sum(axis=1)
    F = F[["GEOID", "ov_km_named"]]
    log("roads done", len(rg))
    save(F, "road")

# Route km, from roads tagged with an Interstate, US or state route number (and Texas Farm to Market roads).
if "route" in STAGES:
    F = pd.DataFrame({"GEOID": gid})
    r = con.sql(f"""
        with seg as (
          select class, geometry, list_transform(coalesce(routes, []), x -> coalesce(x.network, '')) nets
          from '{D}/reference/{R}/{R}-overture-roads.parquet' where subtype='road')
        select class, geometry,
          len(list_filter(nets, n -> n = 'US:I')) > 0 as interstate,
          len(list_filter(nets, n -> n in ('US:I','US:US') or regexp_matches(n, '^US:[A-Z]{{2}}$')
                 or regexp_matches(n, '^US:(I|US|[A-Z]{{2}}):(Business|Loop|Spur|Bypass|Truck|Alternate)'))) > 0 as hwy,
          len(list_filter(nets, n -> regexp_matches(n, '^US:TX:(FM|RM|PR|Spur|Loop|Business)'))) > 0 as txfm
        from seg""").df()
    r = r[r.hwy | r.txfm]
    rg = proj(shapely.from_wkb([bytes(b) for b in r.geometry.values]))
    gi, ti = tree_p.query(rg, predicate="intersects")
    L = shapely.length(shapely.intersection(rg[gi], tg_p[ti])) / 1000
    named = np.isin(r["class"].values[gi], ["motorway", "trunk", "primary", "secondary"])
    for name, m in [("ov_km_route_hwy", r.hwy.values[gi]), ("ov_km_route_hwy_named", r.hwy.values[gi] & named),
                    ("ov_km_route_interstate", r.interstate.values[gi]), ("ov_km_route_txfm", r.txfm.values[gi] & ~r.hwy.values[gi])]:
        F[name] = np.bincount(ti[m], weights=L[m], minlength=n)
    log("routes done", len(r))
    save(F, "route")

if "merge" in STAGES:
    F = pd.DataFrame({"GEOID": gid, "tract_area_km2": shapely.area(tg_p) / 1e6})
    for st in ["bldg", "poi", "road", "route"]:
        F = F.merge(pd.read_parquet(OUT/f"{R}-part-{st}.parquet"), on="GEOID")
    F["region"] = R
    F.to_parquet(OUT/f"{R}-overture-features.parquet", index=False)
    log("done", F.shape)
