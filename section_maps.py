#!/usr/bin/env python3
"""Build a reproducible five-map comparison and a concise HTML/PDF report.

Run from any directory: python reseau_citations/section_maps.py
No network requests. Recomputes five-seed Leiden partitions from local inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import shutil
from itertools import combinations
from pathlib import Path
import textwrap

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import adjusted_rand_score

import analysis as A
from section_groups import SECTION_GROUPS, section_group

LAYERS = [
    ("all", "All body-text citations", None),
    ("history", "History citations", "History"),
    ("culture", "Culture & heritage citations", "Culture & heritage"),
    ("geography", "Geography citations", "Geography"),
    ("administration", "Politics & administration", "Politics & administration"),
]
REPORT = A.HERE / "report"
MAPS = REPORT / "maps"
GAMMA = 0.5


def canonical_mentions(data, nodes, slug_map, title_map):
    m = data["mentions"].copy()
    m["source"] = m.source_slug.map(slug_map)
    m["target"] = m.target_title.map(title_map)
    valid = set(nodes.insee)
    m = m[m.source.isin(valid) & m.target.isin(valid) & m.source.ne(m.target)].copy()
    m["mentions"] = m.n_links + m.n_text
    m = m[m.mentions > 0].copy()
    m["group"] = m.section.map(section_group)
    return m


def layer_edges(m):
    e = m.groupby(["source", "target"], sort=True).agg(
        raw_weight=("mentions", "sum"), n_links=("n_links", "sum"),
        n_text=("n_text", "sum")).reset_index()
    e["narrative_weight"] = 0
    e["log_weight"] = np.log1p(e.raw_weight)
    return e


def cluster_layer(g, gamma=GAMMA):
    runs = []
    for s in range(A.N_SEEDS):
        p = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights="source_normalized",
            resolution_parameter=gamma, seed=A.SEED + s, n_iterations=-1)
        runs.append((p.quality(), np.array(p.membership)))
    membership = A.relabel_by_size(max(runs, key=lambda r: r[0])[1])
    stability = float(np.mean([adjusted_rand_score(a[1], b[1]) for a, b in combinations(runs, 2)]))
    return membership, stability


def weighted_median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind="stable")
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyse_layer(key, title, group, m, nodes, geo, svg=False):
    subset = m if group is None else m[m.group.eq(group)]
    e, u, nd = A.calculate_edge_weights(layer_edges(subset), nodes)
    g, keep = A.lcc_graph(u, nd)
    memb, stability = cluster_layer(g)
    sizes = np.bincount(memb)
    lcc_nodes = nd.set_index("insee").loc[keep].reset_index()
    # Up to 20 readable hub labels. Every community is present in the CSV.
    major = [int(c) for c in np.argsort(-sizes, kind="stable") if sizes[c] >= 50][:20]
    colours = A.adjacency_colouring(memb, lcc_nodes[["x", "y"]].to_numpy() / 1000, major)
    names = {c: lcc_nodes[memb == c].nlargest(1, "in_strength").city.iloc[0] for c in major}
    active = int(nd.strength.gt(0).sum())
    in_lcc = e.source.isin(keep) & e.target.isin(keep)
    meta = dict(gamma=GAMMA, n_nodes=len(nd), W=int(e.raw_weight.sum()), n_edges=len(e),
                n_edges_lcc=int(in_lcc.sum()), title=title,
                subtitle=f"French Wikipedia commune articles · {'all body sections' if group is None else group} only\n"
                         f"{len(keep):,} cities in the largest connected component · Leiden, same settings across maps",
                community_note=f"{len(sizes):,} communities; {len(major)} major hubs labelled. Smaller groups in grey.")
    A.plot_linkedin_map(geo, nd, g, memb, e, colours, names, meta,
                        MAPS / f"{key}.png", MAPS / f"{key}.svg" if svg else None,
                        visual_quantile=0.0)
    Image.open(MAPS / f"{key}.png").convert("RGB").save(MAPS / f"{key}.jpg", quality=91, optimize=True)
    if key == "all":
        render_all_territories(geo, nd, g, memb, colours, names, meta)
    info = nd.set_index("insee")
    d = A.haversine(e.source.map(info.latitude), e.source.map(info.longitude),
                    e.target.map(info.latitude), e.target.map(info.longitude))
    w = e.raw_weight.to_numpy()
    paris = e.source.eq("75056") | e.target.eq("75056")
    memberships = dict(zip(keep, memb.tolist()))
    rows = nd[["insee", "city", "latitude", "longitude", "department", "region",
               "in_strength", "out_strength", "strength"]].copy()
    rows["community"] = rows.insee.map(memberships).astype("Int64")
    rows["status"] = np.where(rows.strength.eq(0), "inactive",
                             np.where(rows.community.notna(), "clustered_lcc", "outside_lcc"))
    rows.to_csv(MAPS / f"{key}_nodes.csv", index=False)
    e.assign(distance_km=d).to_csv(MAPS / f"{key}_edges.csv.gz", index=False)
    stats = dict(key=key, title=title, group=group, nodes=len(nd), active_nodes=active,
                 inactive_nodes=len(nd) - active, lcc_nodes=len(keep), outside_lcc=active - len(keep),
                 lcc_share_active=len(keep) / active, directed_edges=len(e), mentions=int(w.sum()),
                 link_mentions=int(e.n_links.sum()), text_mentions=int(e.n_text.sum()),
                 lcc_share_mentions=float(e.loc[in_lcc, "raw_weight"].sum() / w.sum()),
                 communities=len(sizes), labelled_communities=len(major), gamma=GAMMA,
                 modularity=float(g.modularity(memb.tolist(), weights="source_normalized", resolution=1)),
                 seed_stability_ari=stability, median_distance_km=weighted_median(d, w),
                 share_mentions_under_50km=float(w[d < 50].sum() / w.sum()),
                 median_link_distance_km=weighted_median(d, e.n_links.to_numpy()),
                 paris_endpoint_share=float(w[paris].sum() / w.sum()),
                 top_cited=[dict(city=r.city, mentions=int(r.in_strength))
                            for r in nd.nlargest(5, "in_strength").itertuples()],
                 visual_threshold=meta["visual_threshold"], visual_edges=meta["visual_edges"],
                 visual_share=meta["visual_share"], n_edges_lcc=meta["n_edges_lcc"])
    A.log(f"{key}: {stats['mentions']:,} mentions, {active:,} active cities, "
          f"{len(sizes)} communities; median distance {stats['median_distance_km']:.1f} km")
    return stats, memberships


def method_text():
    return (
        "Corpus: the highest locally stored revision ID for each article, not a live or synchronized Wikipedia snapshot. "
        "Commune IDs come from article INSEE codes validated against cached official reference data; delegated/associated "
        "communes in COG 2025 are folded into their parent. Non-communes, overseas communes and self-citations are removed. "
        "Coordinates use an infobox town centre when within 20 km of the official centre, otherwise the official centre.",
        "A citation is a resolved commune wikilink plus plain-text repeats of that name, anchored by at least one link "
        "anywhere in the article. Ambiguous names, short names, a stoplist and names contained in the source name are excluded "
        "from plain-text matching. Infoboxes, all templates, tables, images, references, comments and end matter are stripped. "
        "This excludes formal reference citations: the network measures mentions of places.",
        "Topics use normalized level-2 headings and a documented ordered keyword list; first match wins and subsections "
        "inherit their parent topic. Culture includes heritage, monuments, sport, tourism and local life; Personalities "
        "is separate. Unmatched sections remain Other. Topic maps are overlapping subsets of the overall network, "
        "not an exhaustive five-way partition. The section inventory lists every heading and its assignment.",
        "For each layer, aggregate w_ij = link counts + plain-text counts, then normalize p_ij = w_ij / sum_j(w_ij). "
        "Cluster the undirected projection p_ij + p_ji using Leiden RBConfigurationVertexPartition. Use fixed gamma = 0.5 "
        "for all five maps, five seeds 20261002–20261006, iterations until no improvement, and retain the highest objective. "
        "Gamma follows the earlier overall-map setting and is not reselected for each topic. Stability is mean pairwise "
        "adjusted Rand index across the ten seed pairs; Q is conventional weighted modularity at gamma = 1, for description.",
        "Each layer is clustered and drawn on its largest weakly connected component. Active cities and citation counts "
        "in the table include all components; coverage columns disclose exclusions. Maps show ties with at least two "
        "mentions (or the maximum if none reach two), but clustering uses every positive tie in that component. "
        "Point sizes and line widths are scaled within each map; compare numeric totals before comparing visual density. "
        "Colours are assigned independently to each map and may repeat. They do not identify the same community across maps. "
        "Up to 20 communities of at least 50 cities are labelled by their most-cited city; other communities are grey. "
        "The overall territories view reuses the exact overall-map memberships and colours; it fits no new clusters. "
        "Hulls show contiguous pockets of at least 12 cities using Delaunay neighbours within 30 km, with a 3.5 km buffer. "
        "Smaller pockets remain dots, so filled areas do not show every member city. These are visual envelopes, not borders.",
        "Distance is the great-circle distance between commune coordinates, weighted by mention count. The links-only "
        "median is a sensitivity check for ambiguous plain-text mentions. Templated prose can survive template removal; "
        "locality may reflect editorial conventions, administrative wording or geography. These are descriptive article "
        "relationships, not evidence of travel, influence or causal historical relationships. ARI between topics is "
        "calculated only on their shared clustered cities and is descriptive; no significance tests are claimed.",
        "Leiden reference: Traag, Waltman & van Eck (2019), From Louvain to Leiden, Scientific Reports 9, 5233, "
        "doi:10.1038/s41598-019-41695-z. Department outlines: france-geojson, simplified IGN Admin Express 2018 "
        "under Licence Ouverte; retrieved 3 October 2026. Label offsets are bounded to 35 km with leader lines; "
        "markers always remain at their original coordinates."
    )


def findings(stats):
    by = {s["key"]: s for s in stats}
    all_, hist, cult, geog, admin = [by[k] for k in ("all", "history", "culture", "geography", "administration")]
    return [
        f"Across the corpus, {all_['mentions']:,} inter-commune mentions connect {all_['active_nodes']:,} cities. "
        f"The median mention spans {all_['median_distance_km']:.1f} km; {all_['share_mentions_under_50km']:.1%} span less than 50 km.",
        f"History has {hist['mentions']:,} mentions with a {hist['median_distance_km']:.1f} km median; "
        f"culture and heritage has {cult['mentions']:,} with a {cult['median_distance_km']:.1f} km median. "
        f"Their largest components cover {hist['lcc_share_active']:.1%} and {cult['lcc_share_active']:.1%} of active cities, respectively.",
        f"Geography's median is {geog['median_distance_km']:.1f} km and administration's is "
        f"{admin['median_distance_km']:.1f} km. The five independently fitted networks yield "
        f"{', '.join(str(s['communities']) for s in stats)} communities, in table order; "
        "network density and connectivity also affect those counts.",
        f"Paris is an endpoint of {all_['paris_endpoint_share']:.1%} of all mentions, "
        f"{hist['paris_endpoint_share']:.1%} in history and {cult['paris_endpoint_share']:.1%} in culture. "
        "This measures textual prominence, not a city's real-world importance."
    ]


def write_report(stats, provenance, comparisons):
    esc = html.escape
    rows = "".join(f"<tr><th>{esc(s['title'])}</th><td>{s['mentions']:,}</td><td>{s['active_nodes']:,}</td>"
                   f"<td>{s['lcc_share_active']:.1%}</td><td>{s['lcc_share_mentions']:.1%}</td>"
                   f"<td>{s['communities']}</td><td>{s['median_distance_km']:.1f}</td>"
                   f"<td>{s['median_link_distance_km']:.1f}</td><td>{s['seed_stability_ari']:.3f}</td></tr>" for s in stats)
    cards = "".join(
        f'<figure id="{s["key"]}"><h2>{esc(s["title"])}</h2><a href="maps/{s["key"]}.png">'
        f'<img src="maps/{s["key"]}.jpg" alt="Geographic citation network: {esc(s["title"])}" loading="lazy"></a>'
        f'<figcaption>{s["directed_edges"]:,} directed pairs; {s["link_mentions"]:,} wikilinks + '
        f'{s["text_mentions"]:,} plain-text repeats. Map: {s["lcc_nodes"]:,} cities; '
        f'{s["visual_edges"]:,} drawn ties (≥{s["visual_threshold"]} mentions). Most-cited: '
        f'{esc(", ".join(t["city"] for t in s["top_cited"][:3]))}. '
        f'<a href="maps/{s["key"]}_nodes.csv">Nodes</a> · '
        f'<a href="maps/{s["key"]}_edges.csv.gz">Edges</a>'
        + (f' · <a href="maps/all_territories.png">Territories: the same {s["communities"]} clusters</a>' if s["key"] == "all" else '')
        + '</figcaption></figure>' for s in stats)
    methods = "".join(f"<p>{esc(t)}</p>" for t in method_text())
    bullets = "".join(f"<li>{esc(t)}</li>" for t in findings(stats))
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Five views of France's citation network</title>
<style>:root{{color-scheme:light}}body{{margin:0;background:#faf7f2;color:#282725;font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1050px;margin:auto;padding:42px 24px}}h1,h2{{font-family:Georgia,serif;font-weight:normal;line-height:1.2}}
h1{{font-size:clamp(30px,5vw,48px);max-width:760px}}.lede{{max-width:760px;color:#625b50;font-size:19px}}
a{{color:#1c5cab}}nav{{display:flex;gap:18px;flex-wrap:wrap;padding:18px 0;border-block:1px solid #e2dacd}}
section{{margin:40px 0}}li{{margin:12px 0}}figure{{margin:40px 0;padding:20px;background:#f3eee5;border:1px solid #e2dacd}}
img{{width:100%;max-width:740px;height:auto;display:block;margin:auto}}figcaption{{font-size:14px;color:#625b50;max-width:740px;margin:15px auto}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:11px 8px;border-bottom:1px solid #e2dacd;text-align:right}}th:first-child{{text-align:left}}
.methods{{max-width:800px}}footer{{font-size:13px;color:#625b50}}@media print{{nav{{display:none}}figure{{break-inside:avoid}}}}
</style></head><body><main><p>WIKIPEDIA · METROPOLITAN FRANCE · LOCAL CORPUS</p>
<h1>Five views of France's citation network</h1><p class="lede">The same map template, five article contexts: where communes mention one another, and how the pattern changes by topic.</p>
<nav><a href="#results">Findings</a><a href="#maps">Maps</a><a href="#method">Method</a><a href="citation_report.pdf">Download PDF</a><a href="section_summary.csv">Data table</a></nav>
<section id="results"><h2>What the maps show</h2><ul>{bullets}</ul><div class="table"><table><thead><tr>
<th>Layer</th><th>Mentions</th><th>Active cities</th><th>City coverage*</th><th>Mention coverage*</th><th>Groups</th><th>Median km</th><th>Links-only km</th><th>Seed ARI</th>
</tr></thead><tbody>{rows}</tbody></table></div><p><small>*Largest-component share of all active cities / all mentions. Counts and distances use all components; groups use the largest component.</small></p></section>
<section id="maps"><h2>The five maps</h2><p>Click a map for its 2400 × 3000 PNG. Colours and point-size scales are local to each map. All maps use the same minimum of two mentions for drawn lines; every positive tie is used in clustering.</p>{cards}</section>
<section id="method" class="methods"><h2>Method and limits</h2>{methods}<p><a href="section_inventory.csv">Section-to-topic inventory</a> · <a href="partition_comparison.csv">Shared-city partition comparisons</a> · <a href="section_report.json">Full numbers and input hashes</a></p>
<p>Algorithm: <a href="https://leidenalg.readthedocs.io/en/stable/reference.html#leidenalg.RBConfigurationVertexPartition">leidenalg reference</a>; Traag, Waltman &amp; van Eck (2019), <a href="https://doi.org/10.1038/s41598-019-41695-z">From Louvain to Leiden</a>. Reference data: cached <a href="https://geo.api.gouv.fr/decoupage-administratif/communes">geo.api.gouv.fr</a>, <a href="https://www.insee.fr/fr/information/8377162">INSEE COG 2025</a>, and <a href="https://github.com/gregoiredavid/france-geojson">france-geojson department outlines</a>.</p></section>
<footer>Prepared 3 October 2026. Article revision IDs and reference file hashes are retained in the exported data. The previous extended report remains in <a href="deep-dive.html">the archived deep dive</a>; its historical numbers were not regenerated for this comparison.</footer>
</main></body></html>'''
    (REPORT / "index.html").write_text(content, encoding="utf-8")
    payload = dict(layers=stats, method=method_text(), findings=findings(stats), provenance=provenance,
                   section_groups=SECTION_GROUPS, partition_comparisons=comparisons)
    (REPORT / "section_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with PdfPages(REPORT / "citation_report.pdf") as pdf:
        text_page(pdf, "Five views of France's citation network", findings(stats) + [
            "Five maps follow. Each uses Leiden with source-normalized weights, gamma 0.5 and five seeds. "
            "Colours and visual scales are independent across maps; line display threshold is two mentions."])
        for s in stats:
            fig = plt.figure(figsize=(8, 10), facecolor=A.BG)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(Image.open(MAPS / f"{s['key']}.jpg"))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        text_page(pdf, "Method: corpus and citations", list(method_text()[:3]), fontsize=10)
        text_page(pdf, "Method: clustering and interpretation", list(method_text()[3:]), fontsize=10)


def text_page(pdf, title, paragraphs, fontsize=11):
    fig = plt.figure(figsize=(8, 10), facecolor=A.BG)
    fig.text(.08, .94, title, fontsize=20, family="Georgia", color=A.INK, va="top")
    y = .85
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=82 if fontsize == 10 else 74)
        fig.text(.08, y, "\n".join(lines), fontsize=fontsize, va="top", color=A.INK2, linespacing=1.5)
        y -= len(lines) * fontsize * 1.5 / 720 + .025
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", action="store_true", help="Also export large SVG maps")
    parser.add_argument("--render-only", action="store_true", help="Re-render exported networks after visual changes; checks data and extraction hashes")
    args = parser.parse_args()
    MAPS.mkdir(parents=True, exist_ok=True)
    data = A.load_data()
    if args.render_only:
        render_saved_maps(data, args.svg)
        return
    nodes, s2i, t2i = A.clean_city_identifiers(data)
    m = canonical_mentions(data, nodes, s2i, t2i)
    inventory = m.groupby(["section", "group"]).agg(
        mentions=("mentions", "sum"), links=("n_links", "sum"), rows=("mentions", "size")).reset_index()
    inventory.to_csv(REPORT / "section_inventory.csv", index=False)
    geo = A.prepare_geodata(data)
    results, memberships = [], {}
    for key, title, group in LAYERS:
        A.log(f"Building {key} map...")
        stats, membership = analyse_layer(key, title, group, m, nodes, geo, args.svg)
        results.append(stats)
        memberships[key] = membership
    comparisons = []
    for a, b in combinations(memberships, 2):
        shared = sorted(set(memberships[a]) & set(memberships[b]))
        comparisons.append(dict(layer_a=a, layer_b=b, common_clustered_nodes=len(shared),
                                ari=float(adjusted_rand_score([memberships[a][c] for c in shared],
                                                              [memberships[b][c] for c in shared]))))
    pd.DataFrame(comparisons).to_csv(REPORT / "partition_comparison.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "top_cited"} for s in results]).to_csv(
        REPORT / "section_summary.csv", index=False)
    provenance = current_provenance(data)
    write_report(results, provenance, comparisons)
    (REPORT / "section_build.log").write_text("\n".join(A.LOG), encoding="utf-8")
    A.log(f"Report written to {REPORT / 'index.html'}")


def current_provenance(data):
    inputs = [A.CACHE / n for n in ("pages.parquet", "mentions.parquet", "geo_communes.json",
                                  "cog_communes_2025.csv", "wiki_redirects.csv")]
    inputs += [A.DEPT_GEOJSON, Path(__file__), A.HERE / "analysis.py",
               A.HERE / "extract_links.py", A.HERE / "section_groups.py"]
    return dict(input_sha256={str(p.relative_to(A.ROOT)).replace("\\", "/"): file_hash(p) for p in inputs},
                      versions={name: importlib.metadata.version(name) for name in (
                          "pandas", "numpy", "igraph", "leidenalg", "matplotlib", "geopandas", "scikit-learn")},
                      seeds=list(range(A.SEED, A.SEED + A.N_SEEDS)), gamma=GAMMA,
                      article_oldid_min=int(data["pages"].oldid.min()),
                      article_oldid_max=int(data["pages"].oldid.max()))


def render_saved_maps(data, svg=False):
    payload = json.loads((REPORT / "section_report.json").read_text(encoding="utf-8"))
    # Geometry/renderer can change, but data and extraction/topic definitions must match.
    for name, expected in payload["provenance"]["input_sha256"].items():
        if Path(name).name in {"pages.parquet", "mentions.parquet", "geo_communes.json",
                              "cog_communes_2025.csv", "wiki_redirects.csv", "extract_links.py", "section_groups.py"}:
            if file_hash(A.ROOT / name) != expected:
                raise ValueError(f"Changed citation input {name}; rebuild without --render-only")
    geo = A.prepare_geodata(data)
    for stats in payload["layers"]:
        key = stats["key"]
        nodes = pd.read_csv(MAPS / f"{key}_nodes.csv", dtype={"insee": str})
        nodes["x"], nodes["y"] = A.TO_L93.transform(nodes.longitude.to_numpy(), nodes.latitude.to_numpy())
        nd = nodes[nodes.community.notna()].copy().sort_values("insee")
        memb = nd.community.to_numpy(dtype=int)
        g = ig.Graph(n=len(nd))
        g.vs["insee"] = nd.insee.tolist()
        sizes = np.bincount(memb)
        major = [int(c) for c in np.argsort(-sizes, kind="stable") if sizes[c] >= 50][:20]
        colours = A.adjacency_colouring(memb, nd[["x", "y"]].to_numpy() / 1000, major)
        names = {c: nd[memb == c].nlargest(1, "in_strength").city.iloc[0] for c in major}
        edges = pd.read_csv(MAPS / f"{key}_edges.csv.gz", dtype={"source": str, "target": str})
        group = stats["group"]
        meta = dict(gamma=stats["gamma"], n_nodes=stats["nodes"], W=stats["mentions"],
                    n_edges=stats["directed_edges"], n_edges_lcc=stats["n_edges_lcc"], title=stats["title"],
                    subtitle=f"French Wikipedia commune articles · {'all body sections' if group is None else group} only\n"
                             f"{len(nd):,} cities in the largest connected component · Leiden, same settings across maps",
                    community_note=f"{len(sizes):,} communities; {len(major)} major hubs labelled. Smaller groups in grey.")
        A.plot_linkedin_map(geo, nodes, g, memb, edges, colours, names, meta,
                            MAPS / f"{key}.png", MAPS / f"{key}.svg" if svg else None, visual_quantile=0.0)
        Image.open(MAPS / f"{key}.png").convert("RGB").save(MAPS / f"{key}.jpg", quality=91, optimize=True)
        if key == "all":
            render_all_territories(geo, nodes, g, memb, colours, names, meta)
        A.log(f"Re-rendered {key}: unchanged city markers, bounded hub labels")
    write_report(payload["layers"], current_provenance(data), payload["partition_comparisons"])


def render_all_territories(geo, nodes, g, memb, colours, names, meta):
    """Alternative geometry for the same partition; never performs clustering."""
    path = MAPS / "all_territories.png"
    A.plot_territories(geo, nodes, g, memb, colours, names, path,
                       meta=dict(gamma=meta["gamma"], title="Citation territories — all body text",
                                 partition_note="Exact same city memberships and colours as the overall citation map."))
    Image.open(path).convert("RGB").save(MAPS / "all_territories.jpg", quality=91, optimize=True)
    membership = [(str(code), int(c)) for code, c in zip(g.vs["insee"], memb)]
    manifest = dict(nodes_file="all_nodes.csv", gamma=meta["gamma"], communities=len(set(memb)),
                    membership_sha256=hashlib.sha256(json.dumps(membership).encode("utf-8")).hexdigest(),
                    colours={str(c): colour for c, colour in colours.items()},
                    minimum_hull_cities=12, spatial_link_km=A.SPATIAL_LINK_KM,
                    hull_buffer_km=3.5, description="Same partition as all.png; contiguous-pocket hulls replace citation lines.")
    (MAPS / "all_territories.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    # Keep the familiar filename aligned with the report, rather than an exploratory partition.
    shutil.copy(path, A.OUT / "citation_territories.png")
    shutil.copy(path, REPORT / "figs" / "citation_territories.png")


if __name__ == "__main__":
    main()
