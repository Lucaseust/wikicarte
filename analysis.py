#!/usr/bin/env python3
"""
Step 3 - Inter-city citation network of metropolitan France.

    citation data -> network -> network communities -> geographic visualisation
                  -> ONLY THEN comparison with départements / régions

Inputs  (built by fetch_reference.py and extract_links.py, see ./cache)
Outputs ./output
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
import re
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import igraph as ig
import leidenalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.spatial import Delaunay, cKDTree  # noqa: E402
from shapely import concave_hull  # noqa: E402
from shapely.geometry import MultiPoint, shape  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, completeness_score,  # noqa: E402
                             homogeneity_score, normalized_mutual_info_score,
                             v_measure_score)

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE
CACHE = HERE / "cache"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
DEPT_GEOJSON = CACHE / "departements_france_metropolitaine.geojson"

SEED = 20261002
RNG = np.random.default_rng(SEED)
RESOLUTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
SELECTABLE_RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # 0.25 / 3.0 exploratory only
N_SEEDS = 5
PRIMARY_WEIGHTING = "source_normalized"   # fixed a priori, see README section of summary
N_PERM = 5000
N_SPIN = 1000
SPATIAL_LINK_KM = 30.0       # Delaunay edges longer than this do not make two cities "contiguous"

TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def strip_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def name_key(t: str) -> str:
    t = strip_accents(str(t)).lower().replace("œ", "oe").replace("æ", "ae")
    t = re.sub(r"[-'’\s]+", " ", t)
    return t.strip()


# =========================================================================== 1. data
def load_data() -> dict:
    pages = pd.read_parquet(CACHE / "pages.parquet")
    mentions = pd.read_parquet(CACHE / "mentions.parquet")
    geo = pd.DataFrame(json.loads((CACHE / "geo_communes.json").read_text(encoding="utf-8")))
    geo["lon_ref"] = geo["centre"].map(lambda c: c["coordinates"][0] if isinstance(c, dict) else np.nan)
    geo["lat_ref"] = geo["centre"].map(lambda c: c["coordinates"][1] if isinstance(c, dict) else np.nan)
    geo = geo.drop(columns="centre").rename(columns={"code": "insee", "nom": "name"})
    cog = pd.read_csv(CACHE / "cog_communes_2025.csv", dtype=str)
    depts = pd.DataFrame(json.loads((CACHE / "geo_departements.json").read_text(encoding="utf-8")))
    regs = pd.DataFrame(json.loads((CACHE / "geo_regions.json").read_text(encoding="utf-8")))
    # plain JSON read (avoids the GDAL dependency of gpd.read_file)
    gj = json.loads(DEPT_GEOJSON.read_text(encoding="utf-8"))
    dept_shapes = gpd.GeoDataFrame([f["properties"] for f in gj["features"]],
                                   geometry=[shape(f["geometry"]) for f in gj["features"]], crs=4326)

    log("=" * 78)
    log("1. DATA INSPECTION — field mapping")
    log("=" * 78)
    log(f"pages parsed (content_wikitext folders)  : {len(pages):,}")
    log(f"body-text mention rows (source,target,section): {len(mentions):,}")
    mapping = {
        "source city": "content_wikitext/<slug>  -> pages.slug -> pages.title (pages_meta.fr_title_canon)",
        "target/mentioned city": "mentions.target_title (wikilink target resolved through article titles + Wikipedia redirects)",
        "number of citations": "mentions.n_links + mentions.n_text (link-anchored mentions; n_links alone kept as sensitivity)",
        "latitude / longitude": "infobox 'latitude'/'longitude' of the article; fallback geo.api.gouv.fr commune centre",
        "INSEE commune code": "infobox 'insee' -> validated against geo.api.gouv.fr (current communes) / INSEE COG 2025 (COMPARENT for former communes)",
        "département": "geo.api.gouv.fr codeDepartement (+ names from /departements)",
        "région": "geo.api.gouv.fr codeRegion (+ names from /regions)",
        "population": "geo.api.gouv.fr population (INSEE légale)",
        "page/text identifiers": "pages.oldid = latest revision id used; pages_meta.wikidata_qid available",
    }
    for k, v in mapping.items():
        log(f"  {k:<24}: {v}")
    return dict(pages=pages, mentions=mentions, geo=geo, cog=cog, depts=depts, regs=regs,
                dept_shapes=dept_shapes)


def clean_city_identifiers(data: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Assign every article a canonical INSEE code; return node table and page->INSEE maps."""
    pages, geo, cog = data["pages"].copy(), data["geo"], data["cog"]
    metro_depts = set(data["dept_shapes"]["code"])
    # pages_meta also lists non-commune articles (départements, régions...): keep commune articles only
    ib = pages["infobox"].fillna("").str.lower().map(strip_accents)
    is_commune = ib.str.contains(r"commune") | (ib.eq("") & pages["insee_infobox"].notna())
    log("")
    log("Infobox types of parsed articles (top 8):")
    for k, v in pages["infobox"].fillna("(none)").value_counts().head(8).items():
        log(f"  {k:<45} {v:>7,}")
    log(f"  -> non-commune articles excluded as nodes: {int((~is_commune).sum()):,}")
    pages = pages[is_commune].copy()
    current = set(geo["insee"])
    geo_by = geo.set_index("insee")
    parent = {r.COM: r.COMPARENT for r in cog.itertuples() if r.TYPECOM in ("COMD", "COMA")
              and isinstance(r.COMPARENT, str)}
    dept_name = dict(zip(data["depts"]["code"], data["depts"]["nom"]))
    # exact (accent/hyphen-insensitive) name index for the rare pages without an infobox code
    name_idx: dict[str, list[str]] = defaultdict(list)
    for code, nm in zip(geo["insee"], geo["name"]):
        name_idx[name_key(nm)].append(code)

    def resolve(row) -> tuple[str | None, str]:
        code = row.insee_infobox
        if isinstance(code, str) and len(code) == 5:
            if code in current:
                return code, "infobox_insee"
            if code in parent and parent[code] in current:
                return parent[code], "former_commune_cog_parent"
        title = str(row.title)
        m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", title)
        base, qual = (m.group(1), m.group(2)) if m else (title, None)
        cands = name_idx.get(name_key(base), [])
        if qual:
            cands = [c for c in cands if name_key(dept_name.get(geo_by.at[c, "codeDepartement"], "")) == name_key(qual)] or \
                    ([] if len(cands) != 1 else cands)
        if len(cands) == 1:
            return cands[0], "exact_name_match(no infobox code)"
        return None, "unresolved"

    res = pages.apply(resolve, axis=1, result_type="expand")
    pages["insee"], pages["id_method"] = res[0], res[1]

    # sanity check: an article's own coordinates must fall near the commune it was mapped to
    lat_r = pages["insee"].map(geo_by["lat_ref"])
    lon_r = pages["insee"].map(geo_by["lon_ref"])
    d = haversine(pages["lat_infobox"], pages["lon_infobox"], lat_r, lon_r)
    pages["coord_gap_km"] = d
    bad_name = (pages["id_method"].str.startswith("exact_name")) & (d > 30)
    pages.loc[bad_name, ["insee", "id_method"]] = [None, "rejected_name_match_far"]

    log("")
    log("Canonical ID = INSEE commune code (COG 2025). Identification method per article:")
    for k, v in pages["id_method"].value_counts().items():
        log(f"  {k:<40} {v:>7,}")

    pages["dept"] = pages["insee"].map(geo_by["codeDepartement"])
    pages["is_metro"] = pages["dept"].isin(metro_depts)
    log(f"  articles outside metropolitan France (overseas) dropped: {int((pages['insee'].notna() & ~pages['is_metro']).sum()):,}")

    keep = pages[pages["is_metro"]].copy()
    # several articles -> one INSEE code (former communes folded into the current one)
    dup = keep["insee"].duplicated(keep=False)
    log(f"  articles folded into an existing INSEE node (former communes): "
        f"{int(dup.sum() - keep.loc[dup, 'insee'].nunique()):,}")
    keep["primary"] = keep["id_method"].eq("infobox_insee")
    primary = keep.sort_values(["primary", "raw_chars"], ascending=False).drop_duplicates("insee")

    nodes = primary[["insee", "title", "slug", "oldid", "lat_infobox", "lon_infobox", "coord_gap_km"]].copy()
    nodes = nodes.merge(geo, on="insee", how="left")
    # infobox coordinates (town centre) when consistent with the official commune centre
    use_ib = nodes["coord_gap_km"].le(20) & nodes["lat_infobox"].notna()
    nodes["latitude"] = np.where(use_ib, nodes["lat_infobox"], nodes["lat_ref"])
    nodes["longitude"] = np.where(use_ib, nodes["lon_infobox"], nodes["lon_ref"])
    log(f"  coordinates from article infobox: {int(use_ib.sum()):,}; from official commune centre: {int((~use_ib).sum()):,}")
    dn = dict(zip(data["depts"]["code"], data["depts"]["nom"]))
    rn = dict(zip(data["regs"]["code"], data["regs"]["nom"]))
    nodes["department"] = nodes["codeDepartement"]
    nodes["department_name"] = nodes["department"].map(dn)
    nodes["region"] = nodes["codeRegion"]
    nodes["region_name"] = nodes["region"].map(rn)
    nodes["old_region"] = nodes["department"].map(DEPT_TO_OLD_REGION)
    nodes["old_region_name"] = nodes["old_region"]
    nodes["city"] = nodes["name"]
    x, y = TO_L93.transform(nodes["longitude"].to_numpy(), nodes["latitude"].to_numpy())
    nodes["x"], nodes["y"] = x, y
    nodes = nodes.sort_values("insee").reset_index(drop=True)
    log(f"  metropolitan city nodes (unique INSEE): {len(nodes):,} "
        f"(official metropolitan communes: {int(geo['codeDepartement'].isin(metro_depts).sum()):,})")

    slug2insee = dict(zip(keep["slug"], keep["insee"]))
    title2insee = dict(zip(keep["title"], keep["insee"]))
    return nodes, slug2insee, title2insee


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


# =========================================================================== 2. network
GEO_ADMIN_KEYS = ("geograph", "urbanism", "localisation", "situation", "climat", "hydrograph",
                  "transport", "voies de communication", "acces", "politique", "administr",
                  "demograph", "population", "enseignement", "equipement", "service", "sante",
                  "intercommunal", "jumelage", "budget", "fiscal", "occupation des sols",
                  "milieux naturels", "risques", "logement", "morphologie")


def section_class(sec: str) -> str:
    if sec == "lead":
        return "lead"
    return "geo_admin" if any(k in sec for k in GEO_ADMIN_KEYS) else "narrative"


def build_network(mentions: pd.DataFrame, nodes: pd.DataFrame, slug2insee: dict, title2insee: dict):
    m = mentions.copy()
    m["source"] = m["source_slug"].map(slug2insee)
    m["target"] = m["target_title"].map(title2insee)
    n0 = len(m)
    valid = set(nodes["insee"])
    m = m[m["source"].isin(valid) & m["target"].isin(valid)]
    self_mask = m["source"] == m["target"]
    log("")
    log("=" * 78)
    log("2. DIRECTED WEIGHTED NETWORK")
    log("=" * 78)
    log(f"mention rows: {n0:,}; with both ends = metropolitan city nodes: {len(m) + 0:,}; "
        f"self-links removed: {int(self_mask.sum()):,}")
    m = m[~self_mask].copy()
    m["mentions"] = m["n_links"] + m["n_text"]
    m["sclass"] = m["section"].map(section_class)
    sec_share = m.groupby("sclass")["mentions"].sum() / m["mentions"].sum()
    log("share of mentions by section class: " + ", ".join(f"{k}={v:.1%}" for k, v in sec_share.items()))
    agg = m.groupby(["source", "target"]).agg(
        raw_weight=("mentions", "sum"), n_links=("n_links", "sum"), n_text=("n_text", "sum")).reset_index()
    narr = m[m["sclass"] == "narrative"].groupby(["source", "target"])["mentions"].sum()
    agg["narrative_weight"] = agg.set_index(["source", "target"]).index.map(narr).fillna(0).astype(int)
    agg["log_weight"] = np.log1p(agg["raw_weight"])
    top_sections = m.groupby("section")["mentions"].sum().sort_values(ascending=False).head(12)
    log("top sections by mentions: " + "; ".join(f"{k} ({v:,})" for k, v in top_sections.items()))
    return agg


def calculate_edge_weights(edges: pd.DataFrame, nodes: pd.DataFrame):
    e = edges.copy()
    out_s = e.groupby("source")["raw_weight"].sum()
    in_s = e.groupby("target")["raw_weight"].sum()
    W = e["raw_weight"].sum()
    e["p_source_norm"] = e["raw_weight"] / e["source"].map(out_s)
    e["assoc_strength"] = e["raw_weight"] * W / (e["source"].map(out_s) * e["target"].map(in_s))
    nodes = nodes.copy()
    nodes["out_strength"] = nodes["insee"].map(out_s).fillna(0).astype(int)
    nodes["in_strength"] = nodes["insee"].map(in_s).fillna(0).astype(int)
    nodes["strength"] = nodes["out_strength"] + nodes["in_strength"]
    nodes["out_degree"] = nodes["insee"].map(e.groupby("source").size()).fillna(0).astype(int)
    nodes["in_degree"] = nodes["insee"].map(e.groupby("target").size()).fillna(0).astype(int)

    # undirected projection w*_ij = w_ij + w_ji (and analogues)
    a = np.where(e["source"] < e["target"], e["source"], e["target"])
    b = np.where(e["source"] < e["target"], e["target"], e["source"])
    u = e.assign(u=a, v=b).groupby(["u", "v"]).agg(
        w=("raw_weight", "sum"), p=("p_source_norm", "sum"),
        narrative=("narrative_weight", "sum"), links=("n_links", "sum"),
        n_dir=("raw_weight", "size")).reset_index()
    s = nodes.set_index("insee")["strength"]
    u["raw"] = u["w"].astype(float)
    u["log"] = np.log1p(u["w"])
    u["source_normalized"] = u["p"]
    u["association"] = u["w"] * (2 * W) / (u["u"].map(s) * u["v"].map(s))
    u["reciprocal"] = u["n_dir"] == 2
    return e, u, nodes


# =========================================================================== 3. descriptives
def describe_network(e: pd.DataFrame, u: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    log("")
    log("=" * 78)
    log("3. DESCRIPTIVE NETWORK STATISTICS (directed, all valid body-text edges)")
    log("=" * 78)
    n = len(nodes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes["insee"])
    G.add_weighted_edges_from(e[["source", "target", "raw_weight"]].itertuples(index=False))
    m = G.number_of_edges()
    W = int(e["raw_weight"].sum())
    recip = nx.reciprocity(G)
    wcc = max(nx.weakly_connected_components(G), key=len)
    scc = max(nx.strongly_connected_components(G), key=len)
    iso = sum(1 for _, d in G.degree() if d == 0)
    st = nodes["strength"]
    log(f"cities (nodes)                    : {n:,}  (isolated, no body-text tie: {iso:,})")
    log(f"directed edges                    : {m:,}")
    log(f"undirected ties                   : {len(u):,}  (reciprocated: {int(u['reciprocal'].sum()):,})")
    log(f"total citation count (mentions)   : {W:,}   [links only: {int(e['n_links'].sum()):,}]")
    log(f"density                           : {m / (n * (n - 1)):.2e}")
    log(f"reciprocity (edge level)          : {recip:.3f}")
    log(f"weighted reciprocity              : {(np.minimum(e['raw_weight'], e.set_index(['source','target']).reindex(list(zip(e['target'], e['source'])))['raw_weight'].fillna(0).to_numpy())).sum() / W:.3f}")
    log(f"largest weakly connected comp.    : {len(wcc):,} nodes ({len(wcc) / n:.1%})")
    log(f"largest strongly connected comp.  : {len(scc):,} nodes ({len(scc) / n:.1%})")
    log(f"weighted degree (in+out) median / mean : {st.median():.0f} / {st.mean():.1f}")
    for col in ("in_strength", "out_strength"):
        q = nodes[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_dict()
        log(f"{col:<12} distribution: " + ", ".join(f"p{int(k * 100)}={v:g}" for k, v in q.items())
            + f", max={nodes[col].max():,}, zero={int((nodes[col] == 0).sum()):,}")
    gini = lambda x: (np.abs(np.subtract.outer(x, x)).mean() / (2 * x.mean())) if len(x) < 5000 else \
        (2 * np.sum(np.arange(1, len(x) + 1) * np.sort(x)) / (len(x) * np.sum(x)) - (len(x) + 1) / len(x))
    log(f"Gini in-strength: {gini(nodes['in_strength'].to_numpy(float)):.3f}; "
        f"Gini out-strength: {gini(nodes['out_strength'].to_numpy(float)):.3f}")
    for col in ("in_strength", "out_strength"):
        top = nodes.nlargest(15, col)
        log(f"top 15 by weighted {col.replace('_strength', '-degree')}: " +
            "; ".join(f"{r.city} ({getattr(r, col):,})" for r in top.itertuples()))

    log("")
    log("Paris dominance check")
    paris = "75056"
    def share(codes):
        codes = set(codes)
        s_src = e.loc[e["source"].isin(codes), "raw_weight"].sum() / W
        s_tgt = e.loc[e["target"].isin(codes), "raw_weight"].sum() / W
        s_any = e.loc[e["source"].isin(codes) | e["target"].isin(codes), "raw_weight"].sum() / W
        e_any = (e["source"].isin(codes) | e["target"].isin(codes)).mean()
        return s_src, s_tgt, s_any, e_any
    s = share([paris])
    log(f"  Paris — as source {s[0]:.2%}, as target {s[1]:.2%}, either endpoint {s[2]:.2%} of citations "
        f"({s[3]:.2%} of directed edges); Paris in-degree = {int(nodes.loc[nodes.insee == paris, 'in_degree'].iloc[0]):,} citing cities "
        f"({nodes.loc[nodes.insee == paris, 'in_degree'].iloc[0] / n:.1%} of all cities)")
    top10 = nodes.nlargest(10, "population")
    s10 = share(top10["insee"])
    log(f"  10 largest cities by population ({', '.join(top10['city'])}):")
    log(f"     as source {s10[0]:.2%}, as target {s10[1]:.2%}, either endpoint {s10[2]:.2%} ({s10[3]:.2%} of edges)")
    in_rank = nodes["in_strength"].rank(ascending=False)
    log(f"  Paris rank by in-strength: {int(in_rank[nodes.insee == paris].iloc[0])}; "
        f"share of Paris among top-1 % in-strength: {nodes.nlargest(int(0.01*n),'in_strength')['in_strength'].sum()/W:.1%} of citations go to top-1 % cities")
    return dict(n=n, m=m, W=W, reciprocity=recip, wcc=len(wcc), paris=s, top10=s10)


# =========================================================================== 4. communities
WEIGHTINGS = ["raw", "log", "source_normalized", "association"]


def lcc_graph(u: pd.DataFrame, nodes: pd.DataFrame) -> tuple[ig.Graph, list[str]]:
    if u.empty:
        raise ValueError("No positive citation ties: cannot cluster an empty network.")
    g_all = nx.Graph()
    g_all.add_edges_from(zip(u["u"], u["v"]))
    lcc = max(nx.connected_components(g_all), key=len)
    keep = sorted(lcc)
    idx = {c: i for i, c in enumerate(keep)}
    uu = u[u["u"].isin(lcc) & u["v"].isin(lcc)]
    g = ig.Graph(n=len(keep), edges=list(zip(uu["u"].map(idx), uu["v"].map(idx))), directed=False)
    for w in WEIGHTINGS:
        g.es[w] = uu[w].to_numpy(float)
    g.vs["insee"] = keep
    return g, keep


def graph_fingerprint(g: ig.Graph) -> str:
    """Bind cached memberships to ordered identifiers, topology, weights and settings."""
    digest = hashlib.sha256()
    digest.update(json.dumps([g.vs["insee"], SEED, N_SEEDS, RESOLUTIONS],
                             ensure_ascii=False).encode("utf-8"))
    digest.update(np.asarray(g.get_edgelist(), dtype="<i8").tobytes())
    for weight in WEIGHTINGS:
        digest.update(weight.encode("ascii"))
        digest.update(np.asarray(g.es[weight], dtype="<f8").tobytes())
    return digest.hexdigest()


def load_partition_cache(g: ig.Graph):
    path = CACHE / "leiden_partitions.pkl"
    stamp = path.with_suffix(".sha256")
    if not stamp.exists() or stamp.read_text().strip() != graph_fingerprint(g):
        raise ValueError("Leiden cache is unverified or stale. Run analysis.py to rebuild it.")
    return pd.read_pickle(path)


def detect_communities(g: ig.Graph) -> tuple[pd.DataFrame, dict]:
    log("")
    log("=" * 78)
    log(f"4. COMMUNITY DETECTION — Leiden (RB-configuration / modularity), {N_SEEDS} seeds per setting, "
        f"undirected projection w*_ij = w_ij + w_ji on the largest connected component ({g.vcount():,} nodes, {g.ecount():,} ties)")
    log("=" * 78)
    rows, parts = [], {}
    for w in WEIGHTINGS:
        weights = g.es[w]
        for gamma in RESOLUTIONS:
            t0 = time.time()
            runs = []
            for s in range(N_SEEDS):
                p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=weights,
                                             resolution_parameter=gamma, seed=SEED + s, n_iterations=-1)
                runs.append((p.quality(), np.array(p.membership)))
            best = max(runs, key=lambda r: r[0])[1]
            best = relabel_by_size(best)
            stab = np.mean([adjusted_rand_score(a[1], b[1]) for a, b in combinations(runs, 2)])
            Q = g.modularity(best.tolist(), weights=weights, resolution=1.0)
            sizes = np.bincount(best)
            rows.append(dict(weighting=w, resolution=gamma, n_communities=len(sizes),
                             n_communities_ge1pct=int((sizes >= 0.01 * len(best)).sum()),
                             modularity=Q, seed_stability_ARI=stab,
                             largest_share=sizes.max() / len(best),
                             sizes_top10=" ".join(map(str, sizes[:10]))))
            parts[(w, gamma)] = best
            log(f"  {w:<18} γ={gamma:<5} k={len(sizes):>4} (≥1%: {rows[-1]['n_communities_ge1pct']:>2})  "
                f"Q={Q:.3f}  stability={stab:.3f}  largest={sizes.max() / len(best):.1%}  [{time.time() - t0:.0f}s]")
    return pd.DataFrame(rows), parts


def relabel_by_size(memb: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.bincount(memb), kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[memb]


def select_primary(sol: pd.DataFrame) -> float:
    """Structural rule fixed before looking at administrative agreement."""
    c = sol[(sol.weighting == PRIMARY_WEIGHTING) & sol.resolution.isin(SELECTABLE_RESOLUTIONS)].copy()
    best = c["seed_stability_ARI"].max()
    c = c[c["seed_stability_ARI"] >= best - 0.02]
    gamma = float(c.sort_values("modularity", ascending=False)["resolution"].iloc[0])
    log("")
    log(f"PRIMARY CLUSTERING (rule fixed a priori): weighting = {PRIMARY_WEIGHTING}; resolution = most seed-stable "
        f"γ in {SELECTABLE_RESOLUTIONS} (ties within 0.02 ARI broken by modularity) -> γ = {gamma}")
    return gamma


# =========================================================================== 5. admin alignment
# Pre-2016 régions (1982-2015), added as a third administrative reference: the 2016 reform merged
# them into today's 13 régions. Static, official INSEE département composition.
OLD_REGIONS = {
    "Alsace": "67 68", "Aquitaine": "24 33 40 47 64", "Auvergne": "03 15 43 63",
    "Basse-Normandie": "14 50 61", "Bourgogne": "21 58 71 89", "Bretagne": "22 29 35 56",
    "Centre": "18 28 36 37 41 45", "Champagne-Ardenne": "08 10 51 52", "Corse": "2A 2B",
    "Franche-Comté": "25 39 70 90", "Haute-Normandie": "27 76",
    "Île-de-France": "75 77 78 91 92 93 94 95", "Languedoc-Roussillon": "11 30 34 48 66",
    "Limousin": "19 23 87", "Lorraine": "54 55 57 88", "Midi-Pyrénées": "09 12 31 32 46 65 81 82",
    "Nord-Pas-de-Calais": "59 62", "Pays de la Loire": "44 49 53 72 85", "Picardie": "02 60 80",
    "Poitou-Charentes": "16 17 79 86", "Provence-Alpes-Côte d'Azur": "04 05 06 13 83 84",
    "Rhône-Alpes": "01 07 26 38 42 69 73 74",
}
DEPT_TO_OLD_REGION = {d: r for r, ds in OLD_REGIONS.items() for d in ds.split()}
ADMIN_LEVELS = {"region": "région (2016)", "old_region": "région (pre-2016)", "department": "département"}


def fast_ari_nmi(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """ARI and NMI (arithmetic normalisation, = sklearn default) from integer-coded labels."""
    ka, kb = a.max() + 1, b.max() + 1
    ct = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    n = ct.sum()
    comb = lambda x: x * (x - 1) / 2
    sa, sb = ct.sum(1), ct.sum(0)
    s_ij, s_a, s_b = comb(ct).sum(), comb(sa).sum(), comb(sb).sum()
    exp = s_a * s_b / comb(n)
    ari = (s_ij - exp) / (0.5 * (s_a + s_b) - exp)
    nz = ct > 0
    pij = ct[nz] / n
    pa, pb = sa / n, sb / n
    mi = np.sum(pij * np.log(pij / np.outer(pa, pb)[nz]))
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    return ari, mi / ((ha + hb) / 2)


def admin_scores(labels, admin) -> dict:
    return dict(ARI=adjusted_rand_score(admin, labels), NMI=normalized_mutual_info_score(admin, labels),
                homogeneity=homogeneity_score(admin, labels), completeness=completeness_score(admin, labels),
                v_measure=v_measure_score(admin, labels))


def evaluate_admin_alignment(sol, parts, g, nodes, primary_key):
    log("")
    log("=" * 78)
    log("5. DO CITATION COMMUNITIES MAP ADMINISTRATIVE TERRITORIES?  (computed after selection)")
    log("=" * 78)
    nd = nodes.set_index("insee").loc[g.vs["insee"]]
    rows = []
    for (w, gamma), memb in parts.items():
        row = dict(weighting=w, resolution=gamma)
        for lvl in ADMIN_LEVELS:
            row.update({f"{k}_{lvl}": v for k, v in admin_scores(memb, nd[lvl].to_numpy()).items()})
        rows.append(row)
    comp = sol.merge(pd.DataFrame(rows), on=["weighting", "resolution"])
    comp["primary"] = [(a, b) == primary_key for a, b in zip(comp.weighting, comp.resolution)]
    dep, reg = nd["department"].to_numpy(), nd["region"].to_numpy()
    ref = admin_scores(dep, reg)
    ref_old = admin_scores(dep, nd["old_region"].to_numpy())
    ref_ro = admin_scores(nd["old_region"].to_numpy(), reg)
    log("Reference (administrative levels compared with each other, same cities):")
    log(f"  départements vs régions 2016   : ARI={ref['ARI']:.3f}, NMI={ref['NMI']:.3f}")
    log(f"  départements vs pre-2016 régions: ARI={ref_old['ARI']:.3f}, NMI={ref_old['NMI']:.3f}")
    log(f"  pre-2016 régions vs régions 2016: ARI={ref_ro['ARI']:.3f}, NMI={ref_ro['NMI']:.3f}")
    log(f"  number of units: {nd['department'].nunique()} départements, {nd['old_region'].nunique()} pre-2016 régions, "
        f"{nd['region'].nunique()} régions")
    memb = parts[primary_key]
    p = comp[comp.primary].iloc[0]
    log(f"Primary solution: {int(p.n_communities)} communities")
    for lvl, lab in ADMIN_LEVELS.items():
        log(f"  vs {lab:<18}: ARI={p[f'ARI_{lvl}']:.3f}  NMI={p[f'NMI_{lvl}']:.3f}  homogeneity={p[f'homogeneity_{lvl}']:.3f}  "
            f"completeness={p[f'completeness_{lvl}']:.3f}  V={p[f'v_measure_{lvl}']:.3f}")
    log("  (homogeneity = each community lies within one unit; completeness = each unit lies within one community)")

    ct_reg = pd.crosstab(memb, nd["region_name"].to_numpy())
    ct_dep = pd.crosstab(memb, (nd["department"] + " " + nd["department_name"]).to_numpy())
    ct_old = pd.crosstab(memb, nd["old_region"].to_numpy())
    ct_reg.index.name = ct_dep.index.name = ct_old.index.name = "community"
    per = []
    for c in ct_reg.index:
        rr, dd, oo = ct_reg.loc[c], ct_dep.loc[c], ct_old.loc[c]
        per.append(dict(community=c, size=int(rr.sum()), dominant_region=rr.idxmax(),
                        pct_dominant_region=rr.max() / rr.sum(), n_regions=int((rr > 0).sum()),
                        dominant_old_region=oo.idxmax(), pct_dominant_old_region=oo.max() / oo.sum(),
                        dominant_department=dd.idxmax(), pct_dominant_department=dd.max() / dd.sum(),
                        n_departments=int((dd > 0).sum())))
    per = pd.DataFrame(per)
    # reverse view: how concentrated is each administrative unit in one community?
    reg_conc = (ct_reg.max(axis=0) / ct_reg.sum(axis=0)).sort_values()
    dep_conc = (ct_dep.max(axis=0) / ct_dep.sum(axis=0))
    log(f"Share of each région's cities in its main community: median {reg_conc.median():.1%}, "
        f"min {reg_conc.min():.1%} ({reg_conc.index[0]}), max {reg_conc.max():.1%}")
    log(f"Share of each département's cities in its main community: median {dep_conc.median():.1%}; "
        f"départements with ≥90% in one community: {int((dep_conc >= 0.9).sum())}/{len(dep_conc)}")
    old_conc = (ct_old.max(axis=0) / ct_old.sum(axis=0)).sort_values()
    log(f"Share of each pre-2016 région's cities in its main community: median {old_conc.median():.1%}, "
        f"min {old_conc.min():.1%} ({old_conc.index[0]}), max {old_conc.max():.1%}")
    log(f"Communities whose cities are ≥90% in one région: {int((per.pct_dominant_region >= 0.9).sum())}/{len(per)}; "
        f"in one pre-2016 région: {int((per.pct_dominant_old_region >= 0.9).sum())}/{len(per)}; "
        f"in one département: {int((per.pct_dominant_department >= 0.9).sum())}/{len(per)}")
    nd_comm_per_dep = (ct_dep > 0).sum(axis=0)
    log(f"Median number of communities touching a département: {nd_comm_per_dep.median():.0f}; "
        f"median départements per community: {per.n_departments.median():.0f}")
    return comp, ct_reg, ct_dep, ct_old, per, ref


# =========================================================================== 6. permutation tests
def run_permutation_test(memb, g, nodes) -> pd.DataFrame:
    log("")
    log("=" * 78)
    log(f"6. PERMUTATION TESTS ({N_PERM:,} label permutations; {N_SPIN:,} spatial 'spin' rotations)")
    log("=" * 78)
    nd = nodes.set_index("insee").loc[g.vs["insee"]]
    xy = nd[["x", "y"]].to_numpy() / 1000.0
    rng = np.random.default_rng(SEED)
    out = []
    # sanity check of the fast implementation against scikit-learn
    a0 = pd.factorize(nd["region"])[0]
    assert abs(fast_ari_nmi(a0, memb)[0] - adjusted_rand_score(a0, memb)) < 1e-9
    assert abs(fast_ari_nmi(a0, memb)[1] - normalized_mutual_info_score(a0, memb)) < 1e-9
    for lvl in ADMIN_LEVELS:
        admin = pd.factorize(nd[lvl])[0]
        obs_ari, obs_nmi = fast_ari_nmi(admin, memb)
        null_ari = np.empty(N_PERM)
        null_nmi = np.empty(N_PERM)
        for i in range(N_PERM):
            null_ari[i], null_nmi[i] = fast_ari_nmi(rng.permutation(admin), memb)
        # spatial null: rotate the whole community map around France's centroid (+ random mirror),
        # each city takes the community of the nearest rotated city. Preserves the size, shape and
        # spatial autocorrelation of communities, destroys their alignment with real boundaries.
        c = xy.mean(axis=0)
        spin_ari = np.empty(N_SPIN)
        spin_nmi = np.empty(N_SPIN)
        for i in range(N_SPIN):
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            if rng.random() < 0.5:
                R = R @ np.array([[-1, 0], [0, 1]])
            rot = (xy - c) @ R.T + c
            _, j = cKDTree(rot).query(xy)
            spin_ari[i], spin_nmi[i] = fast_ari_nmi(admin, memb[j])
        for metric, obs, nulls in (("ARI", obs_ari, (null_ari, spin_ari)), ("NMI", obs_nmi, (null_nmi, spin_nmi))):
            for name, nl in zip(("label_permutation", "spatial_spin"), nulls):
                p = (1 + np.sum(nl >= obs)) / (len(nl) + 1)
                z = (obs - nl.mean()) / (nl.std() + 1e-12)
                out.append(dict(level=lvl, metric=metric, null=name, observed=obs, null_mean=nl.mean(),
                                null_sd=nl.std(), null_p95=np.quantile(nl, 0.95), null_max=nl.max(),
                                z=z, p_empirical=p))
                log(f"  {lvl:<10} {metric}: observed={obs:.3f} | {name:<17} null mean={nl.mean():.3f} "
                    f"(95th pct {np.quantile(nl, 0.95):.3f}, max {nl.max():.3f}) -> z={z:.1f}, p={p:.4f}")
    log("  NOTE: label permutation destroys spatial autocorrelation and is therefore anti-conservative;")
    log("  the spin null keeps communities spatially compact and asks whether they line up with the *actual*")
    log("  administrative boundaries better than an equally compact but arbitrarily placed partition.")
    return pd.DataFrame(out)


# =========================================================================== 7. spatial statistics
def spatial_adjacency(xy_km: np.ndarray) -> np.ndarray:
    tri = Delaunay(xy_km)
    s = tri.simplices
    pairs = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    d = np.linalg.norm(xy_km[pairs[:, 0]] - xy_km[pairs[:, 1]], axis=1)
    return pairs[d <= SPATIAL_LINK_KM]


def calculate_spatial_statistics(memb, g, nodes, e, names) -> tuple[pd.DataFrame, dict]:
    log("")
    log("=" * 78)
    log("7. SPATIAL COHERENCE")
    log("=" * 78)
    nd = nodes.set_index("insee").loc[g.vs["insee"]].reset_index()
    xy = nd[["x", "y"]].to_numpy() / 1000.0
    rng = np.random.default_rng(SEED + 7)
    n = len(nd)
    adj = spatial_adjacency(xy)
    rows = []
    for c in np.unique(memb):
        idx = np.where(memb == c)[0]
        pts = xy[idx]
        sub = idx if len(idx) <= 1500 else rng.choice(idx, 1500, replace=False)
        pd_ = np.linalg.norm(xy[sub][:, None] - xy[sub][None], axis=2)[np.triu_indices(len(sub), 1)]
        cen = pts.mean(axis=0)
        r = np.linalg.norm(pts - cen, axis=1)
        mask = np.isin(adj[:, 0], idx) & np.isin(adj[:, 1], idx)
        sg = nx.Graph()
        sg.add_nodes_from(idx)
        sg.add_edges_from(map(tuple, adj[mask]))
        comps = sorted((len(cc) for cc in nx.connected_components(sg)), reverse=True)
        lon, lat = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(cen[0] * 1000, cen[1] * 1000)
        rows.append(dict(community=c, name=names.get(c, ""), size=len(idx),
                         median_pairwise_km=float(np.median(pd_)) if len(pd_) else 0.0,
                         centroid_lat=lat, centroid_lon=lon, radius_median_km=float(np.median(r)),
                         radius_p90_km=float(np.quantile(r, 0.9)), n_pockets=len(comps),
                         n_pockets_ge5=int(sum(1 for s in comps if s >= 5)),
                         share_main_pocket=comps[0] / len(idx)))
    cs = pd.DataFrame(rows)

    # within vs between community distances
    i = rng.integers(0, n, 400_000)
    j = rng.integers(0, n, 400_000)
    ok = i != j
    i, j = i[ok], j[ok]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    same = memb[i] == memb[j]
    res = dict(within_mean=d[same].mean(), within_median=np.median(d[same]),
               between_mean=d[~same].mean(), between_median=np.median(d[~same]), random_mean=d.mean(),
               random_median=np.median(d))
    log(f"random city pairs: mean {res['random_mean']:.0f} km (median {res['random_median']:.0f})")
    log(f"same-community pairs: mean {res['within_mean']:.0f} km (median {res['within_median']:.0f}); "
        f"different-community pairs: mean {res['between_mean']:.0f} km (median {res['between_median']:.0f})")
    big = cs[cs["size"] >= 0.01 * n]
    log(f"major communities (≥1% of cities, n={len(big)}): median radius {big.radius_median_km.median():.0f} km; "
        f"median share in main contiguous pocket {big.share_main_pocket.median():.1%}; "
        f"median pockets with ≥5 cities {big.n_pockets_ge5.median():.0f}")

    # edge distances
    pos = nodes.set_index("insee")[["x", "y"]] / 1000.0
    ex = e.join(pos, on="source").join(pos, on="target", rsuffix="_t")
    ed = np.hypot(ex["x"] - ex["x_t"], ex["y"] - ex["y_t"]).to_numpy()
    e = e.assign(distance_km=ed)
    res.update(edge_mean=ed.mean(), edge_median=np.median(ed),
               edge_wmean=np.average(ed, weights=e["raw_weight"]),
               share_edges_lt50=(ed < 50).mean(), share_w_lt50=e.loc[ed < 50, "raw_weight"].sum() / e["raw_weight"].sum())
    allxy = nodes[["x", "y"]].to_numpy() / 1000.0
    ii, jj = rng.integers(0, len(allxy), 500_000), rng.integers(0, len(allxy), 500_000)
    rd = np.linalg.norm(allxy[ii] - allxy[jj], axis=1)
    rd = rd[ii != jj]
    log(f"citation edges: mean length {ed.mean():.0f} km (median {np.median(ed):.0f}), citation-weighted mean "
        f"{res['edge_wmean']:.0f} km; random pairs mean {rd.mean():.0f} km (median {np.median(rd):.0f})")
    log(f"  {res['share_edges_lt50']:.1%} of edges and {res['share_w_lt50']:.1%} of citations link cities < 50 km apart "
        f"(random pairs: {(rd < 50).mean():.2%})")

    # distance decay: probability of a tie by distance band (pair counts estimated from random pairs)
    bins = np.array([0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1300])
    n_all = len(allxy) * (len(allxy) - 1)
    pair_frac = np.histogram(rd, bins)[0] / len(rd)
    edge_cnt = np.histogram(ed, bins)[0]
    decay = pd.DataFrame(dict(lo=bins[:-1], hi=bins[1:], n_edges=edge_cnt,
                              est_pairs=pair_frac * n_all))
    decay["p_tie"] = decay["n_edges"] / decay["est_pairs"].replace(0, np.nan)
    decay["mean_weight"] = [e.loc[(ed >= lo) & (ed < hi), "raw_weight"].mean() for lo, hi in zip(bins[:-1], bins[1:])]
    log("P(directed tie) by distance: " + "; ".join(
        f"{int(r.lo)}-{int(r.hi)} km: {r.p_tie:.1e}" for r in decay.itertuples() if r.n_edges > 0))

    # gravity-type regression with source fixed effects (within transformation)
    #   log(1 + w_ij) ~ log d_ij + log pop_j  | source FE ; case-control: all edges + 10 random non-ties per source
    popl = np.log1p(nodes.set_index("insee")["population"].fillna(0))
    codes = nodes["insee"].to_numpy()
    src_with = e["source"].unique()
    neg_s = np.repeat(src_with, 10)
    neg_t = codes[rng.integers(0, len(codes), len(neg_s))]
    neg = pd.DataFrame(dict(source=neg_s, target=neg_t, raw_weight=0))
    neg = neg[neg.source != neg.target]
    neg = neg[~neg.set_index(["source", "target"]).index.isin(e.set_index(["source", "target"]).index)]
    df = pd.concat([e[["source", "target", "raw_weight"]], neg], ignore_index=True)
    df = df.join(pos, on="source").join(pos, on="target", rsuffix="_t")
    df["logd"] = np.log(np.maximum(np.hypot(df.x - df.x_t, df.y - df.y_t), 1.0))
    df["logpop_t"] = df["target"].map(popl)
    df["y"] = np.log1p(df["raw_weight"])
    df["tie"] = (df["raw_weight"] > 0).astype(float)
    for col in ("y", "tie", "logd", "logpop_t"):
        df[col + "_dm"] = df[col] - df.groupby("source")[col].transform("mean")
    reg_res = {}
    for dep_var in ("y", "tie"):
        X = df[["logd_dm", "logpop_t_dm"]]
        fit = sm.OLS(df[dep_var + "_dm"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["source"])[0]})
        reg_res[dep_var] = fit
        log(f"  within-source OLS  {'log(1+w)' if dep_var == 'y' else 'tie (LPM)'} ~ log d + log pop_target: "
            f"β_logd = {fit.params['logd_dm']:.3f} (SE {fit.bse['logd_dm']:.4f}), β_logpop = {fit.params['logpop_t_dm']:.3f} "
            f"(SE {fit.bse['logpop_t_dm']:.4f}); n = {int(fit.nobs):,}, clustered by source")
    ed_only = df[df.raw_weight > 0]
    fit = sm.OLS(ed_only["y_dm"], ed_only[["logd_dm", "logpop_t_dm"]]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(ed_only["source"])[0]})
    log(f"  intensive margin (existing ties only): β_logd = {fit.params['logd_dm']:.3f} (SE {fit.bse['logd_dm']:.4f})")
    res["decay"] = decay
    res["beta_logd_y"] = reg_res["y"].params["logd_dm"]
    return cs, res, e


# =========================================================================== 8-10. figures
BG = "#faf7f2"
INK = "#1f1d1a"
INK2 = "#5c5850"
INK3 = "#8f8a80"
OUTLINE = "#b9b1a3"
DEPT_LINE = "#e4ddd1"
REG_LINE = "#c9bfae"
# Validated categorical hues (dataviz reference palette) + 2 earthy extensions. Colours are assigned by
# graph-colouring the geographic adjacency of communities, so neighbours never share a hue; identity is
# carried by direct hub labels, not by colour alone.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
           "#8a5a44", "#3f8f9b"]
MINOR = "#bdb6aa"

plt.rcParams.update({
    "font.family": "Segoe UI", "svg.fonttype": "none", "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG,
})


def prepare_geodata(data: dict) -> dict:
    dep = data["dept_shapes"].to_crs(2154)
    dmap = data["cog"].query("TYPECOM == 'COM'").drop_duplicates("DEP").set_index("DEP")["REG"]
    dep["region"] = dep["code"].map(dmap)
    reg = dep.dissolve("region").reset_index()
    rn = dict(zip(data["regs"]["code"], data["regs"]["nom"]))
    reg["region_name"] = reg["region"].map(rn)
    dep["old_region"] = dep["code"].map(DEPT_TO_OLD_REGION)
    old = dep.dissolve("old_region").reset_index()
    outline = gpd.GeoSeries([dep.union_all()], crs=2154)
    return dict(dep=dep, reg=reg, old=old, outline=outline)


def adjacency_colouring(memb, xy_km, major: list[int]) -> dict[int, str]:
    # broad notion of "neighbour" for colouring: any city within another community's 25 nearest
    # neighbours, so near-touching communities (e.g. across a thin third community) get distinct hues
    k = min(26, len(xy_km))
    if k < 2:
        return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(major)}
    _, nn = cKDTree(xy_km).query(xy_km, k=k)
    adj = np.column_stack([np.repeat(np.arange(len(xy_km)), k - 1), nn[:, 1:].ravel()])
    cnt = Counter()
    for a, b in adj:
        ca, cb = memb[a], memb[b]
        if ca != cb:
            cnt[(min(ca, cb), max(ca, cb))] += 1
    nb = defaultdict(Counter)
    for (a, b), k in cnt.items():
        nb[a][b] += k
        nb[b][a] += k
    lab = {c: np.array(to_rgb(h)) for c, h in zip(range(len(PALETTE)), PALETTE)}
    colours: dict[int, int] = {}
    usage = Counter()
    for c in major:  # largest first
        best, best_score = None, -1e9
        for k in range(len(PALETTE)):
            conflict = sum(v for n_, v in nb[c].items() if colours.get(n_) == k)
            # perceptual-ish distance to coloured neighbours (rgb proxy), weighted by shared border
            dmin = min([np.linalg.norm(lab[k] - lab[colours[n_]]) for n_ in nb[c] if n_ in colours] or [2.0])
            score = -1000 * conflict + dmin - 0.15 * usage[k]
            if score > best_score:
                best, best_score = k, score
        colours[c] = best
        usage[best] += 1
    return {c: PALETTE[k] for c, k in colours.items()}


def bezier(p0, p1, bend=0.18, n=12):
    """Quadratic curve bending to the right of the direction p0 -> p1 (direction encoded subtly)."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    d = p1 - p0
    ctrl = (p0 + p1) / 2 + bend * np.column_stack([d[:, 1], -d[:, 0]])
    t = np.linspace(0, 1, n)[None, :, None]
    return (1 - t) ** 2 * p0[:, None] + 2 * (1 - t) * t * ctrl[:, None] + t ** 2 * p1[:, None]


def blend(c1, c2, a):
    return tuple(np.array(to_rgb(c1)) * (1 - a) + np.array(to_rgb(c2)) * a)


def draw_base(ax, geo, dept=True, reg_lw=0.5, reg_color=REG_LINE):
    geo["outline"].plot(ax=ax, facecolor="#f3eee5", edgecolor="none", zorder=0)
    if dept:
        geo["dep"].boundary.plot(ax=ax, color=DEPT_LINE, linewidth=0.22, zorder=1)
    geo["reg"].boundary.plot(ax=ax, color=reg_color, linewidth=reg_lw, zorder=1)
    geo["outline"].boundary.plot(ax=ax, color=OUTLINE, linewidth=0.6, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def node_sizes(strength, smin=0.08, smax=32.0, power=1.7):
    """Area grows with sqrt(strength) (convex-scaled), clamped at the 99.95th percentile so Paris stays sane."""
    s = np.sqrt(np.asarray(strength, float))
    if not len(s):
        return s
    cap = np.quantile(s, 0.9995)
    if cap <= 0:
        return np.full(len(s), smin)
    s = np.clip(s / cap, 0, 1)
    return smin + (smax - smin) * s ** power


def constrain_labels(ax, texts, anchors, max_distance=35_000):
    """Keep geographical hub labels within 35 km of their actual city marker.

    Label optimization controls overlap, never the location of a city. Leader
    lines are drawn only after positions have been constrained.
    """
    for text, anchor in zip(texts, anchors):
        anchor = np.asarray(anchor, dtype=float)
        delta = np.asarray(text.get_position()) - anchor
        distance = np.linalg.norm(delta)
        if distance > max_distance:
            delta *= max_distance / distance
            text.set_position(anchor + delta)
        if np.linalg.norm(delta) > 2_000:
            ax.annotate("", xy=anchor, xytext=text.get_position(),
                        arrowprops=dict(arrowstyle="-", color=INK3, lw=0.45), zorder=5)


def plot_linkedin_map(geo, nodes, g, memb, e, colours, names, meta, path_png, path_svg=None,
                      visual_quantile=0.85, full=False):
    nd = nodes.set_index("insee").loc[g.vs["insee"]].reset_index()
    comm_of = dict(zip(nd["insee"], memb))
    pos = nd.set_index("insee")[["x", "y"]]
    figsize = (8, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.1, 0.96, 0.765])
    draw_base(ax, geo)

    ee = e[e["source"].isin(comm_of) & e["target"].isin(comm_of)].copy()
    if ee.empty:
        plt.close(fig)
        raise ValueError("No citation ties to render in this component.")
    if full:
        vis = ee
        thr = int(ee["raw_weight"].min())
    else:
        thr = max(2, int(np.quantile(ee["raw_weight"], visual_quantile)))
        vis = ee[ee["raw_weight"] >= thr]
        if vis.empty:
            thr = int(ee["raw_weight"].max())
            vis = ee[ee["raw_weight"] >= thr]
    vis = vis.sort_values("raw_weight")
    meta["visual_threshold"] = thr
    meta["visual_edges"] = len(vis)
    meta["visual_share"] = len(vis) / len(ee)
    p0 = pos.loc[vis["source"]].to_numpy()
    p1 = pos.loc[vis["target"]].to_numpy()
    curves = bezier(p0, p1, bend=0.16)
    lw = np.log1p(vis["raw_weight"].to_numpy())
    lwn = (lw - lw.min()) / (lw.max() - lw.min() + 1e-9)
    cs = vis["source"].map(comm_of).to_numpy()
    ct = vis["target"].map(comm_of).to_numpy()
    recip = vis.set_index(["source", "target"]).index.isin(
        pd.MultiIndex.from_arrays([ee["target"], ee["source"]]))
    if full:
        alpha = 0.02 + 0.16 * lwn
        widths = 0.08 + 0.7 * lwn
    else:
        alpha = 0.07 + 0.45 * lwn ** 0.8 + 0.05 * recip
        widths = 0.18 + 1.5 * lwn ** 1.1
    cols = []
    for a_, b_, al in zip(cs, ct, alpha):
        base = colours.get(a_, MINOR) if a_ == b_ else blend(colours.get(a_, MINOR), "#7d776d", 0.6)
        cols.append((*to_rgb(base), min(al, 0.58)))
    ax.add_collection(LineCollection(curves, colors=cols, linewidths=widths, capstyle="round", zorder=2))

    order = np.argsort(nd["strength"].to_numpy())
    sz = node_sizes(nd["strength"].to_numpy())[order]
    nc = [colours.get(c, MINOR) for c in memb[order]]
    big = sz >= np.sort(sz)[-min(60, len(sz))]
    ax.scatter(nd["x"].to_numpy()[order][~big], nd["y"].to_numpy()[order][~big], s=sz[~big],
               c=[nc[i] for i in np.where(~big)[0]], linewidths=0, alpha=0.72, zorder=3, rasterized=True)
    ax.scatter(nd["x"].to_numpy()[order][big], nd["y"].to_numpy()[order][big], s=sz[big],
               c=[nc[i] for i in np.where(big)[0]], edgecolors=BG, linewidths=0.5, alpha=0.97, zorder=4)

    # labels: the hub (most-cited city) of every community — the labels double as the legend
    hubs = {nd[memb == c].nlargest(1, "in_strength")["insee"].iloc[0]: c for c in colours}
    texts = []
    anchors = []
    smax = nd["strength"].max()
    for code, c in hubs.items():
        r = nd[nd.insee == code].iloc[0]
        fs = 6.3 + 2.4 * min(1.0, math.sqrt(r.strength / smax) * 1.5)
        t = ax.text(r.x, r.y, r.city, fontsize=fs, color=INK, ha="left", va="bottom", zorder=6,
                    fontweight="semibold" if fs > 7.6 else "normal")
        t.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground=BG, alpha=0.9)])
        texts.append(t)
        anchors.append((r.x, r.y))
    obst = nd.nlargest(400, "strength")
    xmin, ymin, xmax, ymax = geo["outline"].total_bounds
    pad = 15_000
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    adjust_text(texts, ax=ax, x=obst["x"].to_numpy(), y=obst["y"].to_numpy(),
                expand=(1.15, 1.3), force_text=(0.35, 0.5), force_static=(0.15, 0.25),
                force_pull=(0.04, 0.04), max_move=(8, 8), iter_lim=400)
    constrain_labels(ax, texts, anchors)

    title = meta.get("title", "How French cities cite one another" if not full else "How French cities cite one another — every tie")
    fig.text(0.05, 0.955, title, fontsize=21, family="Georgia", color=INK, va="top")
    fig.text(0.05, 0.915, meta.get("subtitle", "Network communities derived from body-text mentions between the French Wikipedia\n"
             "pages of metropolitan communes — largest connected component"),
             fontsize=9.6, color=INK2, va="top", linespacing=1.45)
    fig.text(0.05, 0.868, "Node size = citation activity  ·  Colour = network community  ·  Line strength = citation frequency",
             fontsize=7.6, color=INK3, va="top")

    fig.text(0.05, 0.075, meta.get("community_note", f"{len(colours)} major communities labelled by their most-cited city; smaller groups in grey."),
             fontsize=7.4, color=INK2, va="bottom")
    vis_note = (f"All {len(ee):,} directed ties in the largest connected component shown." if full else
                f"Lines: the {meta['visual_share']:.0%} strongest directed ties (≥{thr} mentions, {meta['visual_edges']:,} of "
                f"{meta['n_edges_lcc']:,}); all ties are used in the analysis.")
    fig.text(0.05, 0.014,
             f"{vis_note}\nCurves bend clockwise from citing to cited city. Communities: Leiden on source-normalised "
             f"ties, γ = {meta['gamma']}.\nData: French Wikipedia, body text of {meta['n_nodes']:,} commune articles "
             f"(infobox, templates, tables, references and navigation excluded) · {meta['W']:,} mentions.",
             fontsize=6.1, color=INK3, va="bottom", linespacing=1.5)
    fig.savefig(path_png, dpi=300)
    if path_svg:
        fig.savefig(path_svg)
    plt.close(fig)


def label_units(ax, gdf, col):
    for r in gdf.itertuples():
        p = r.geometry.representative_point()
        nm = getattr(r, col)
        nm = nm.replace("-", "-\n", 1) if len(nm) > 18 else nm
        t = ax.text(p.x, p.y, nm, fontsize=6.0, color=INK, ha="center", va="center", zorder=7, style="italic")
        t.set_path_effects([patheffects.withStroke(linewidth=2, foreground=BG, alpha=0.85)])


def plot_admin_comparison(geo, nodes, g, memb, colours, names, admin_row, path):
    nd = nodes.set_index("insee").loc[g.vs["insee"]].reset_index()
    xy_km = nd[["x", "y"]].to_numpy() / 1000
    sz = np.clip(node_sizes(nd["strength"].to_numpy(), 0.4, 30), 0.4, 30)
    order = np.argsort(sz)

    def unit_colours(col):
        codes = sorted(nd[col].unique())
        idx = np.array([codes.index(v) for v in nd[col]])
        cmap = adjacency_colouring(idx, xy_km, list(range(len(codes))))
        return [cmap[i] for i in idx]

    comm_cols = [colours.get(c, MINOR) for c in memb]
    fig, axes = plt.subplots(2, 2, figsize=(14, 15.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.885, bottom=0.04, wspace=0.02, hspace=0.06)
    panels = [
        ("A   Citation communities", comm_cols, "comm", None),
        ("B   Administrative régions (2016–)", unit_colours("region"), "reg", None),
        ("C   Pre-2016 régions (1982–2015)", unit_colours("old_region"), "old", None),
        ("D   Communities over régions (dark) and départements (pale)", comm_cols, "overlay", None),
    ]
    for ax, (ttl, col, kind, _) in zip(axes.flat, panels):
        geo["outline"].plot(ax=ax, facecolor="#f3eee5", edgecolor="none", zorder=0)
        if kind == "overlay":
            geo["dep"].boundary.plot(ax=ax, color="#cfc6b6", linewidth=0.35, zorder=4)
            geo["reg"].boundary.plot(ax=ax, color="#3d3933", linewidth=1.1, zorder=4)
        elif kind == "reg":
            geo["reg"].boundary.plot(ax=ax, color="#6f685d", linewidth=0.8, zorder=4)
        elif kind == "old":
            geo["old"].boundary.plot(ax=ax, color="#6f685d", linewidth=0.8, zorder=4)
        geo["outline"].boundary.plot(ax=ax, color=OUTLINE, linewidth=0.6, zorder=1)
        ax.scatter(nd["x"].to_numpy()[order], nd["y"].to_numpy()[order], s=sz[order],
                   c=np.array(col, dtype=object)[order].tolist(), linewidths=0, alpha=0.85, zorder=3,
                   rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, loc="left", fontsize=12.5, color=INK, family="Georgia", pad=4)
        if kind == "reg":
            label_units(ax, geo["reg"], "region_name")
        elif kind == "old":
            label_units(ax, geo["old"], "old_region")
        else:
            texts = []
            anchors = []
            for c in colours:
                hub = nd[memb == c].nlargest(1, "in_strength").iloc[0]
                t = ax.text(hub.x, hub.y, names[c], fontsize=6.2, color=INK, zorder=7)
                t.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground=BG, alpha=0.9)])
                texts.append(t)
                anchors.append((hub.x, hub.y))
            adjust_text(texts, ax=ax, expand=(1.1, 1.2), force_pull=(0.04, 0.04), max_move=(8, 8), iter_lim=200)
            constrain_labels(ax, texts, anchors)
    fig.text(0.012, 0.975, "Do citation communities follow administrative boundaries?", fontsize=21,
             family="Georgia", color=INK, va="top")
    a = admin_row
    fig.text(0.012, 0.945,
             f"{int(a['n_communities'])} communities (Leiden, source-normalised ties, γ = {a['resolution']}).   Agreement — "
             f"régions 2016: ARI {a['ARI_region']:.2f} / NMI {a['NMI_region']:.2f}   ·   pre-2016 régions: "
             f"ARI {a['ARI_old_region']:.2f} / NMI {a['NMI_old_region']:.2f}   ·   départements: "
             f"ARI {a['ARI_department']:.2f} / NMI {a['NMI_department']:.2f}",
             fontsize=10, color=INK2, va="top")
    fig.text(0.012, 0.925, "Communities are built only from citation ties; administrative labels were never used to "
             "build or select them. Each dot is a commune.", fontsize=9, color=INK3, va="top")
    fig.text(0.012, 0.012, "Colours in A and D: network communities; in B and C: administrative units. Within each "
             "panel neighbouring units never share a colour, so colours are not comparable across panels.",
             fontsize=8, color=INK3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_territories(geo, nodes, g, memb, colours, names, path, meta=None):
    meta = meta or {}
    nd = nodes.set_index("insee").loc[g.vs["insee"]].reset_index()
    xy = nd[["x", "y"]].to_numpy()
    adj = spatial_adjacency(xy / 1000)
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.035, 0.06, 0.93, 0.8])
    draw_base(ax, geo, dept=False, reg_lw=0.7, reg_color="#9d9589")
    for c, col in colours.items():
        idx = np.where(memb == c)[0]
        mask = np.isin(adj[:, 0], idx) & np.isin(adj[:, 1], idx)
        sg = nx.Graph()
        sg.add_nodes_from(idx)
        sg.add_edges_from(map(tuple, adj[mask]))
        for comp in nx.connected_components(sg):
            if len(comp) < 12:  # isolated pockets are shown as dots only, never bridged
                continue
            pts = MultiPoint(xy[list(comp)])
            hull = concave_hull(pts, ratio=0.12).buffer(3500).simplify(800)
            hull = hull.intersection(geo["outline"].iloc[0])
            gpd.GeoSeries([hull], crs=2154).plot(ax=ax, facecolor=col, alpha=0.30, edgecolor=col,
                                                  linewidth=0.6, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.25, c=[colours.get(c, MINOR) for c in memb], linewidths=0, alpha=0.6,
               zorder=3, rasterized=True)
    texts = []
    anchors = []
    for c in colours:
        hub = nd[memb == c].nlargest(1, "in_strength").iloc[0]
        t = ax.text(hub.x, hub.y, names[c], fontsize=7, color=INK, zorder=6, fontweight="semibold")
        t.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground=BG, alpha=0.9)])
        texts.append(t)
        anchors.append((hub.x, hub.y))
    xmin, ymin, xmax, ymax = geo["outline"].total_bounds
    ax.set_xlim(xmin - 15000, xmax + 15000)
    ax.set_ylim(ymin - 15000, ymax + 15000)
    adjust_text(texts, ax=ax, force_pull=(0.04, 0.04), max_move=(8, 8), iter_lim=200)
    constrain_labels(ax, texts, anchors)
    fig.text(0.05, 0.955, meta.get("title", "Citation territories"), fontsize=21, family="Georgia", color=INK, va="top")
    description = "Network communities drawn as hulls around contiguous pockets; régions shown for reference."
    if "gamma" in meta:
        description = (f"{len(np.unique(memb))} communities · Leiden, source-normalised ties · γ = {meta['gamma']}\n"
                       + meta.get("partition_note", description))
    fig.text(0.05, 0.915, description,
             fontsize=9.2, color=INK2, va="top", linespacing=1.45)
    fig.text(0.05, 0.02, f"Hulls drawn separately for each spatially contiguous pocket of ≥12 cities (Delaunay links ≤ "
             f"{SPATIAL_LINK_KM:.0f} km); smaller detached pockets appear as dots only.", fontsize=6.4, color=INK3)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_distance_decay(decay: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    mid = np.sqrt(np.maximum(decay["lo"], 1) * decay["hi"])
    ok = decay["n_edges"] > 0
    ax.plot(mid[ok], decay["p_tie"][ok], color=PALETTE[0], lw=2, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance between cities (km)", color=INK2)
    ax.set_ylabel("Probability that A cites B", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(OUTLINE)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color="#ebe5da", lw=0.6)
    ax.set_title("Citations decay steeply with distance", loc="left", family="Georgia", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# =========================================================================== 11. outputs
def save_outputs(nodes, g, memb, names, comp, ct_reg, ct_dep, ct_old, per, cs, perm, e, u):
    nd = nodes.copy()
    cm = dict(zip(g.vs["insee"], memb))
    nd["community"] = nd["insee"].map(cm).astype("Int64")
    nd["community_name"] = nd["community"].map(lambda c: names.get(c, "") if pd.notna(c) else "")
    cols = ["city", "insee", "latitude", "longitude", "population", "in_strength", "out_strength",
            "community", "community_name", "department", "department_name", "region", "region_name", "old_region",
            "in_degree", "out_degree", "title", "oldid"]
    nd.rename(columns={"insee": "insee"})[cols].rename(columns={"insee": "insee_code"}).to_csv(
        OUT / "citation_communities.csv", index=False, encoding="utf-8")
    comp.to_csv(OUT / "community_admin_comparison.csv", index=False, encoding="utf-8", float_format="%.4f")
    ct_reg.to_csv(OUT / "contingency_community_x_region.csv", encoding="utf-8")
    ct_dep.to_csv(OUT / "contingency_community_x_department.csv", encoding="utf-8")
    ct_old.to_csv(OUT / "contingency_community_x_pre2016_region.csv", encoding="utf-8")
    per.assign(name=per["community"].map(names)).merge(cs.drop(columns=["name", "size"]), on="community").to_csv(
        OUT / "community_profiles.csv", index=False, encoding="utf-8", float_format="%.4f")
    perm.to_csv(OUT / "permutation_tests.csv", index=False, float_format="%.5f")
    e.to_csv(OUT / "edges_directed.csv.gz", index=False, encoding="utf-8", float_format="%.6g")


# =========================================================================== main
def main() -> None:
    t0 = time.time()
    data = load_data()
    nodes, slug2insee, title2insee = clean_city_identifiers(data)
    edges = build_network(data["mentions"], nodes, slug2insee, title2insee)
    e, u, nodes = calculate_edge_weights(edges, nodes)
    desc = describe_network(e, u, nodes)

    g, keep = lcc_graph(u, nodes)
    cache_parts = CACHE / "leiden_partitions.pkl"
    if "--reuse" in sys.argv and cache_parts.exists():
        sol, parts, saved_log = load_partition_cache(g)
        LOG.extend(saved_log)
        print("\n".join(saved_log))
    else:
        n0 = len(LOG)
        sol, parts = detect_communities(g)
        pd.to_pickle((sol, parts, LOG[n0:]), cache_parts)
        cache_parts.with_suffix(".sha256").write_text(graph_fingerprint(g))
    gamma = select_primary(sol)
    key = (PRIMARY_WEIGHTING, gamma)
    memb = parts[key]

    # how much do the weightings agree with each other (network-only robustness)
    log("")
    log("Agreement between weightings (ARI between partitions at the selected γ, and narrative-only network):")
    for w in WEIGHTINGS:
        if w != PRIMARY_WEIGHTING:
            log(f"  {PRIMARY_WEIGHTING} vs {w:<12}: ARI = {adjusted_rand_score(memb, parts[(w, gamma)]):.3f}")
    # narrative-section-only network (excludes lead and geography/administration sections)
    un = u[u["narrative"] > 0]
    idx = {c: i for i, c in enumerate(g.vs["insee"])}
    un = un[un["u"].isin(idx) & un["v"].isin(idx)]
    gn = ig.Graph(n=g.vcount(), edges=list(zip(un["u"].map(idx), un["v"].map(idx))), directed=False)
    out_n = e[e.narrative_weight > 0].groupby("source")["narrative_weight"].sum()
    en = e[e.narrative_weight > 0]
    pn = en.assign(p=en.narrative_weight / en.source.map(out_n))
    a = np.minimum(pn.source, pn.target)
    b = np.maximum(pn.source, pn.target)
    pw = pn.assign(u=a, v=b).groupby(["u", "v"])["p"].sum()
    gn.es["w"] = pw.reindex(list(zip(un["u"], un["v"]))).to_numpy()
    pn_part = leidenalg.find_partition(gn, leidenalg.RBConfigurationVertexPartition, weights="w",
                                       resolution_parameter=gamma, seed=SEED, n_iterations=-1)
    narr_memb = np.array(pn_part.membership)
    in_narr = np.array(gn.degree()) > 0
    log(f"  narrative-only network ({len(un):,} ties, {in_narr.sum():,} cities): ARI with primary = "
        f"{adjusted_rand_score(memb[in_narr], narr_memb[in_narr]):.3f}")

    # community names = most-cited city (in-strength) of each community
    nd = nodes.set_index("insee").loc[g.vs["insee"]].reset_index()
    names = {}
    for c in np.unique(memb):
        sub = nd[memb == c]
        names[c] = sub.nlargest(1, "in_strength")["city"].iloc[0]
    sizes = np.bincount(memb)
    log("")
    log("Primary communities (size, hub, top-5 cities by in-strength):")
    for c in np.argsort(-sizes)[:30]:
        sub = nd[memb == c].nlargest(5, "in_strength")
        log(f"  C{c:<3} n={sizes[c]:>5}  {names[c]:<24} {', '.join(sub['city'])}")

    comp, ct_reg, ct_dep, ct_old, per, ref = evaluate_admin_alignment(sol, parts, g, nodes, key)
    # narrative robustness vs admin
    narr_scores = admin_scores(narr_memb[in_narr], nd["region"].to_numpy()[in_narr])
    narr_scores_d = admin_scores(narr_memb[in_narr], nd["department"].to_numpy()[in_narr])
    narr_scores_o = admin_scores(narr_memb[in_narr], nd["old_region"].to_numpy()[in_narr])
    log(f"Narrative-only network ({len(np.unique(narr_memb[in_narr]))} communities): pre-2016 régions ARI={narr_scores_o['ARI']:.3f} NMI={narr_scores_o['NMI']:.3f}")
    log(f"Narrative-only network vs régions ARI={narr_scores['ARI']:.3f} NMI={narr_scores['NMI']:.3f}; "
        f"vs départements ARI={narr_scores_d['ARI']:.3f} NMI={narr_scores_d['NMI']:.3f}")
    log("")
    log("All solutions (weighting × γ):")
    log(comp[["weighting", "resolution", "n_communities", "n_communities_ge1pct", "modularity", "seed_stability_ARI",
              "ARI_region", "NMI_region", "ARI_old_region", "NMI_old_region", "ARI_department", "NMI_department"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    log("")
    log("Per-community administrative profile (primary):")
    pp = per.assign(name=per.community.map(names)).sort_values("size", ascending=False).head(30)
    for r in pp.itertuples():
        log(f"  C{r.community:<3} {r.name:<22} n={r.size:>5}  région: {r.dominant_region:<28} {r.pct_dominant_region:>6.1%} "
            f"({r.n_regions} rég.)  old: {r.dominant_old_region:<20} {r.pct_dominant_old_region:>6.1%}  dépt: {r.dominant_department:<26} {r.pct_dominant_department:>6.1%} ({r.n_departments} dép.)")

    perm = run_permutation_test(memb, g, nodes)
    cs, sres, e = calculate_spatial_statistics(memb, g, nodes, e, names)

    # ---------------------------------------------------------------- figures
    geo = prepare_geodata(data)
    xy_km = nd[["x", "y"]].to_numpy() / 1000
    major = [c for c in np.argsort(-sizes) if sizes[c] >= 50]
    colours = adjacency_colouring(memb, xy_km, major)
    names_major = {c: names[c] for c in major}
    meta = dict(gamma=gamma, n_nodes=len(nodes), W=desc["W"], n_edges=desc["m"],
                n_edges_lcc=int((e.source.isin(set(keep)) & e.target.isin(set(keep))).sum()))
    plot_linkedin_map(geo, nodes, g, memb, e, colours, names_major, meta,
                      OUT / "france_city_citation_network_linkedin.png",
                      OUT / "france_city_citation_network_linkedin.svg")
    log(f"visual threshold (LinkedIn map): ≥{meta['visual_threshold']} mentions -> {meta['visual_edges']:,} directed ties "
        f"({meta['visual_share']:.1%} of ties in the LCC); analysis uses all ties")
    meta_full = dict(meta)
    plot_linkedin_map(geo, nodes, g, memb, e, colours, names_major, meta_full,
                      OUT / "france_city_citation_network_full.png", full=True)
    prow = comp[comp.primary].iloc[0]
    plot_admin_comparison(geo, nodes, g, memb, colours, names_major, prow, OUT / "citation_communities_vs_regions.png")
    plot_territories(geo, nodes, g, memb, colours, names_major, OUT / "citation_territories_extended.png",
                     meta=dict(gamma=gamma, title="Citation territories — extended analysis"))
    plot_distance_decay(sres["decay"], OUT / "distance_decay.png")

    save_outputs(nodes, g, memb, names, comp, ct_reg, ct_dep, ct_old, per, cs, perm, e, u)
    log("")
    log(f"done in {time.time() - t0:.0f}s")
    (OUT / "network_summary.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
