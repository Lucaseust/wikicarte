#!/usr/bin/env python3
"""
Step 4 - "Who cites whom, and how": descriptive deep-dive for the report.

Reuses the cleaned network from analysis.py (same identifiers, same body-text definition,
same cached Leiden partition) and writes figures + a JSON of numbers to ./report.
"""

from __future__ import annotations

import json
import math
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import analysis as A  # noqa: E402

REPORT = A.HERE / "report"
FIGS = REPORT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S: dict = {}          # numbers quoted in the report
T: dict = {}          # small tables quoted in the report

INK, INK2, INK3, BG = A.INK, A.INK2, A.INK3, A.BG
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

SIZE_BINS = [0, 200, 500, 2_000, 10_000, 50_000, 200_000, 10**8]
SIZE_LABELS = ["<200", "200–500", "500–2k", "2k–10k", "10k–50k", "50k–200k", "200k+"]

from section_groups import SECTION_GROUPS, section_group  # noqa: E402, F401


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(A.OUTLINE)
    ax.tick_params(colors=INK2, labelsize=8.5)
    if grid:
        ax.grid(True, axis=grid, color="#ebe5da", lw=0.6)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIGS / name, dpi=200)
    plt.close(fig)


def halo(t, w=2.4):
    t.set_path_effects([patheffects.withStroke(linewidth=w, foreground=BG, alpha=0.9)])


# =============================================================================== load
def load():
    data = A.load_data()
    nodes, s2i, t2i = A.clean_city_identifiers(data)
    edges = A.build_network(data["mentions"], nodes, s2i, t2i)
    e, u, nodes = A.calculate_edge_weights(edges, nodes)
    g, keep = A.lcc_graph(u, nodes)
    sol, parts, _ = A.load_partition_cache(g)
    gamma = A.select_primary(sol)
    memb = parts[(A.PRIMARY_WEIGHTING, gamma)]
    nodes["community"] = nodes["insee"].map(dict(zip(g.vs["insee"], memb)))

    # section-level mention table (same filters as build_network)
    m = data["mentions"].copy()
    m["source"], m["target"] = m["source_slug"].map(s2i), m["target_title"].map(t2i)
    valid = set(nodes["insee"])
    m = m[m.source.isin(valid) & m.target.isin(valid) & (m.source != m.target)].copy()
    m["mentions"] = m["n_links"] + m["n_text"]
    m["group"] = m["section"].map(section_group)

    # article length (primary article of each node)
    pages = data["pages"].set_index("slug")
    nodes["body_chars"] = nodes["slug"].map(pages["body_chars"])
    nodes["raw_chars"] = nodes["slug"].map(pages["raw_chars"])

    # administrative seat status (INSEE COG 2025)
    reg = pd.read_csv(A.CACHE / "v_region_2025.csv", dtype=str)
    dep = pd.read_csv(A.CACHE / "v_departement_2025.csv", dtype=str)
    arr = pd.read_csv(A.CACHE / "v_arrondissement_2025.csv", dtype=str)
    seat = pd.Series("Ordinary commune", index=nodes["insee"].values)
    seat[seat.index.isin(arr["CHEFLIEU"])] = "Sub-prefecture"
    seat[seat.index.isin(dep["CHEFLIEU"])] = "Prefecture"
    seat[seat.index.isin(reg["CHEFLIEU"])] = "Regional capital"
    nodes["seat"] = nodes["insee"].map(seat)
    nodes["size_class"] = pd.cut(nodes["population"].fillna(0), SIZE_BINS, right=False, labels=SIZE_LABELS)

    pos = nodes.set_index("insee")[["x", "y"]] / 1000
    e = e.join(pos, on="source").join(pos, on="target", rsuffix="_t")
    e["dist"] = np.hypot(e.x - e.x_t, e.y - e.y_t)
    e = e.drop(columns=["x", "y", "x_t", "y_t"])
    m = m.merge(e[["source", "target", "dist"]], on=["source", "target"], how="left")
    info = nodes.set_index("insee")
    for side, col in (("s", "source"), ("t", "target")):
        e[f"pop_{side}"] = e[col].map(info["population"])
        e[f"seat_{side}"] = e[col].map(info["seat"])
        e[f"class_{side}"] = e[col].map(info["size_class"])
        e[f"comm_{side}"] = e[col].map(info["community"])
    return data, nodes, e, u, m, g, memb, gamma


# =============================================================================== sections
def overview(nodes, e, m):
    W = int(e.raw_weight.sum())
    S.update(n_cities=len(nodes), n_edges=len(e), W=W, n_links=int(e.n_links.sum()), n_text=int(e.n_text.sum()),
             share_links=float(e.n_links.sum() / W),
             cited_by_any=int((nodes.in_strength > 0).sum()), cites_any=int((nodes.out_strength > 0).sum()),
             median_out_degree=float(nodes.out_degree.median()), median_in_degree=float(nodes.in_degree.median()),
             mean_out_degree=float(nodes.out_degree.mean()),
             pct_weight_1=float((e.raw_weight == 1).mean()), max_weight=int(e.raw_weight.max()))


def who_cites(nodes):
    n = nodes[nodes.body_chars > 0].copy()
    S["rho_out_body"] = float(spearmanr(n.out_strength, n.body_chars).statistic)
    S["rho_out_pop"] = float(spearmanr(n.out_strength, n.population).statistic)
    S["rho_body_pop"] = float(spearmanr(n.body_chars, n.population).statistic)
    n["per10k"] = n.out_strength / (n.body_chars / 10_000)
    S["median_per10k"] = float(n.loc[n.body_chars >= 2000, "per10k"].median())
    S["median_body_chars"] = float(n.body_chars.median())
    X = sm.add_constant(np.column_stack([np.log(n.body_chars), np.log1p(n.population.fillna(0))]))
    fit = sm.OLS(np.log1p(n.out_strength), X).fit(cov_type="HC1")
    S["out_elast_body"], S["out_elast_pop"], S["out_r2"] = map(float, (fit.params.iloc[1], fit.params.iloc[2], fit.rsquared))
    X1 = sm.add_constant(np.log(n.body_chars))
    S["out_r2_body_only"] = float(sm.OLS(np.log1p(n.out_strength), X1).fit().rsquared)
    T["top_citers"] = n.nlargest(10, "out_strength")[["city", "department_name", "out_strength", "out_degree",
                                                     "body_chars", "population"]].to_dict("records")
    by_class = n.groupby("size_class", observed=True).agg(
        out_median=("out_strength", "median"), body_median=("body_chars", "median"),
        outdeg_median=("out_degree", "median"), n=("insee", "size"))
    T["out_by_class"] = by_class.reset_index().to_dict("records")

    # figure: article length vs citations made
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    sub = n.sample(min(len(n), 12000), random_state=1)
    ax.scatter(sub.body_chars / 1000, sub.out_strength + 1, s=2, c=BLUE, alpha=0.18, linewidths=0, rasterized=True)
    bins = np.logspace(np.log10(0.3), np.log10(n.body_chars.max() / 1000), 22)
    mid = np.sqrt(bins[:-1] * bins[1:])
    grp = [n.loc[(n.body_chars / 1000 >= a) & (n.body_chars / 1000 < b), "out_strength"] for a, b in zip(bins[:-1], bins[1:])]
    keep = [len(x) >= 30 for x in grp]
    mid = mid[keep]
    med = [x.median() + 1 for x, k in zip(grp, keep) if k]
    ax.plot(mid, med, color=INK, lw=1.8)
    ax.text(mid[-6], med[-6] * 1.5, "median", color=INK, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Length of the article's body text (thousand characters)", color=INK2, fontsize=9)
    ax.set_ylabel("Mentions of other cities (+1)", color=INK2, fontsize=9)
    style(ax, "both")
    ax.set_title("Longer articles cite more cities", loc="left", family="Georgia", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "who_cites_length.png")


def who_is_cited(nodes, e):
    n = nodes[nodes.population > 0].copy()
    W = e.raw_weight.sum()
    S["n_never_cited"] = int((nodes.in_strength == 0).sum())
    S["pop_median_never_cited"] = float(nodes.loc[nodes.in_strength == 0, "population"].median())
    S["pop_median_cited"] = float(nodes.loc[nodes.in_strength > 0, "population"].median())
    top1 = nodes.nlargest(int(0.01 * len(nodes)), "in_strength")["in_strength"].sum() / W
    top10c = nodes.nlargest(10, "in_strength")["in_strength"].sum() / W
    S["share_top1pct_in"] = float(top1)
    S["share_top10_in"] = float(top10c)
    S["rho_in_pop"] = float(spearmanr(n.in_strength, n.population).statistic)

    # gravity of fame: log1p(in) ~ log pop (+ seat dummies)
    y = np.log1p(n.in_strength)
    X0 = sm.add_constant(np.log(n.population))
    f0 = sm.OLS(y, X0).fit(cov_type="HC1")
    S["in_elast_pop"], S["in_r2_pop"] = float(f0.params.iloc[1]), float(f0.rsquared)
    d = pd.get_dummies(n.seat)[["Sub-prefecture", "Prefecture", "Regional capital"]].astype(float)
    X1 = sm.add_constant(pd.concat([np.log(n.population).rename("logpop"), d], axis=1))
    f1 = sm.OLS(y, X1).fit(cov_type="HC1")
    S["seat_effects"] = {k: dict(beta=float(f1.params[k]), mult=float(math.exp(f1.params[k])), se=float(f1.bse[k]))
                         for k in d.columns}
    S["in_r2_pop_seat"] = float(f1.rsquared)
    seat_tab = nodes.groupby("seat").agg(n=("insee", "size"), in_median=("in_strength", "median"),
                                         in_total=("in_strength", "sum"), pop_median=("population", "median"))
    seat_tab["share_of_citations"] = seat_tab.in_total / W
    seat_tab["share_of_cities"] = seat_tab.n / len(nodes)
    T["seat"] = seat_tab.reset_index().to_dict("records")

    # over / under-cited relative to population
    n["expected"] = np.expm1(f0.predict(X0))
    n["ratio"] = (n.in_strength + 1) / (n.expected + 1)
    over = n[n.in_strength >= 150].nlargest(15, "ratio")
    under = n[n.population >= 30_000].nsmallest(15, "ratio")
    cols = ["city", "department_name", "population", "in_strength", "expected", "ratio", "seat"]
    T["over_cited"] = over[cols].to_dict("records")
    T["under_cited"] = under[cols].to_dict("records")
    T["top_cited"] = nodes.nlargest(15, "in_strength")[["city", "department_name", "population", "in_strength",
                                                        "in_degree", "seat"]].to_dict("records")

    # figure: in-strength vs population
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {"Ordinary commune": "#c9c1b3", "Sub-prefecture": "#86b6ef", "Prefecture": BLUE,
              "Regional capital": "#0d366b"}
    for k in ["Ordinary commune", "Sub-prefecture", "Prefecture", "Regional capital"]:
        s = n[n.seat == k]
        ax.scatter(s.population, s.in_strength + 1, s=3 if k == "Ordinary commune" else 14, c=colors[k],
                   alpha=0.35 if k == "Ordinary commune" else 0.9, linewidths=0, label=k,
                   rasterized=k == "Ordinary commune", zorder=2 if k == "Ordinary commune" else 3)
    xs = np.logspace(np.log10(n.population.min()), np.log10(n.population.max()), 50)
    ax.plot(xs, np.expm1(f0.params.iloc[0] + f0.params.iloc[1] * np.log(xs)) + 1, color=INK, lw=1.2, ls="--", zorder=4)
    # a few hand-picked labels with fixed offsets (left/right of the point)
    picks = {"Paris": "l", "Lyon": "l", "Arras": "l", "Foix": "l", "Saint-Pierre-de-Frugie": "r",
             "Licques": "r", "Mont-Louis": "r", "Vitrolles": "l", "Herblay-sur-Seine": "r", "Villepinte": "l"}
    for city, side in picks.items():
        r = n[n.city == city]
        if r.empty:
            continue
        r = r.nlargest(1, "population").iloc[0]
        ax.scatter([r.population], [r.in_strength + 1], s=22, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=5)
        t = ax.text(r.population * (1.25 if side == "r" else 0.8), r.in_strength + 1, city, fontsize=7.6, color=INK,
                    ha="left" if side == "r" else "right", va="center", zorder=6)
        halo(t)
    xr = 2.5e2
    ax.text(xr, np.expm1(f0.params.iloc[0] + f0.params.iloc[1] * np.log(xr)) * 0.55, "expected from population",
            fontsize=7.6, color=INK2, rotation=17, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Population (INSEE)", color=INK2, fontsize=9)
    ax.set_ylabel("Mentions received from other city articles (+1)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2, markerscale=1.6)
    style(ax, "both")
    ax.set_title("Bigger cities are cited more, but administrative seats even more",
                 loc="left", family="Georgia", fontsize=12.5, color=INK)
    fig.tight_layout()
    save(fig, "who_is_cited_population.png")


def hierarchy(nodes, e):
    W = e.raw_weight.sum()
    ok = e.pop_s.notna() & e.pop_t.notna()
    up = e.loc[ok & (e.pop_t > e.pop_s), "raw_weight"].sum() / e.loc[ok, "raw_weight"].sum()
    S["share_up"] = float(up)
    S["share_up_10x"] = float(e.loc[ok & (e.pop_t > 10 * e.pop_s), "raw_weight"].sum() / e.loc[ok, "raw_weight"].sum())
    S["share_down_10x"] = float(e.loc[ok & (e.pop_s > 10 * e.pop_t), "raw_weight"].sum() / e.loc[ok, "raw_weight"].sum())
    F = e.pivot_table(index="class_s", columns="class_t", values="raw_weight", aggfunc="sum", observed=False).fillna(0)
    F = F.reindex(index=SIZE_LABELS, columns=SIZE_LABELS).fillna(0)
    exp = np.outer(F.sum(axis=1), F.sum(axis=0)) / F.values.sum()
    ratio = F / exp
    share = F.div(F.sum(axis=1), axis=0)
    T["class_share"] = share.round(4).to_dict()
    # net balance per class
    cls = nodes.groupby("size_class", observed=False).agg(out=("out_strength", "sum"), inn=("in_strength", "sum"),
                                                          n=("insee", "size"))
    cls["net"] = cls.inn - cls.out
    cls["in_per_city"] = cls.inn / cls.n
    cls["out_per_city"] = cls.out / cls.n
    cls["share_cities"] = cls.n / cls.n.sum()
    cls["share_in"] = cls.inn / W
    cls["share_out"] = cls.out / W
    T["class_balance"] = cls.reset_index().astype({"size_class": str}).to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    lr = np.log2(ratio.values)
    ax.imshow(lr, cmap=LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"]),
                   vmin=-3, vmax=3)
    for i in range(len(SIZE_LABELS)):
        for j in range(len(SIZE_LABELS)):
            v = ratio.values[i, j]
            ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(lr[i, j]) > 2 else INK)
    ax.set_xticks(range(7), SIZE_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(7), SIZE_LABELS)
    ax.set_xlabel("Population of the CITED city", color=INK2, fontsize=9)
    ax.set_ylabel("Population of the CITING city", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.3, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Citations observed ÷ expected from activity alone", loc="left", family="Georgia", fontsize=11.5,
                 color=INK)
    ax = axes[1]
    yy = np.arange(len(SIZE_LABELS))
    ax.barh(yy + 0.2, cls.share_out.values * 100, height=0.38, color="#c9c1b3", label="share of citations made")
    ax.barh(yy - 0.2, cls.share_in.values * 100, height=0.38, color=BLUE, label="share of citations received")
    ax.set_yticks(yy, SIZE_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("% of all mentions", color=INK2, fontsize=9)
    style(ax, "x")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    ax.set_title("Small places write, big places are written about", loc="left", family="Georgia", fontsize=11.5,
                 color=INK)
    fig.tight_layout()
    save(fig, "hierarchy.png")


def reciprocity(nodes, e):
    key = set(zip(e.source, e.target))
    e["recip"] = [(t, s) in key for s, t in zip(e.source, e.target)]
    S["reciprocity"] = float(e.recip.mean())
    bands = [0, 5, 10, 20, 50, 100, 200, 2000]
    e["dband"] = pd.cut(e.dist, bands, right=False)
    rb = e.groupby("dband", observed=True)["recip"].mean()
    T["recip_by_distance"] = {str(k): float(v) for k, v in rb.items()}
    ok = e.pop_s.notna() & e.pop_t.notna() & (e.pop_s > 0) & (e.pop_t > 0)
    lr = np.abs(np.log10(e.loc[ok, "pop_t"] / e.loc[ok, "pop_s"]))
    rr = e.loc[ok].groupby(pd.cut(lr, [0, 0.25, 0.5, 1, 1.5, 2, 10], right=False), observed=True)["recip"].mean()
    T["recip_by_popratio"] = {str(k): float(v) for k, v in rr.items()}
    # strongest mutual pairs
    w = e.set_index(["source", "target"])["raw_weight"]
    mut = e[e.recip & (e.source < e.target)].copy()
    mut["back"] = [w[(t, s)] for s, t in zip(mut.source, mut.target)]
    mut["mutual"] = np.minimum(mut.raw_weight, mut.back)
    name = nodes.set_index("insee")["city"]
    top = mut.nlargest(12, "mutual")
    T["mutual_pairs"] = [dict(a=name[r.source], b=name[r.target], ab=int(r.raw_weight), ba=int(r.back),
                              dist=float(r.dist)) for r in top.itertuples()]
    # asymmetry with big cities: small->big vs big->small for pairs with 10x population gap
    S["recip_10x"] = float(e.loc[ok & (e.pop_t > 10 * e.pop_s), "recip"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    labels = ["0–5", "5–10", "10–20", "20–50", "50–100", "100–200", "200+"]
    ax.bar(range(len(rb)), rb.values * 100, color=BLUE, width=0.62)
    ax.set_xticks(range(len(rb)), labels[:len(rb)])
    ax.set_xlabel("Distance between the two cities (km)", color=INK2, fontsize=9)
    ax.set_ylabel("% of ties that are reciprocated", color=INK2, fontsize=9)
    style(ax)
    ax.set_title("Neighbours cite each other back", loc="left", family="Georgia", fontsize=11.5, color=INK)
    ax = axes[1]
    plabels = ["<1.8×", "1.8–3×", "3–10×", "10–30×", "30–100×", ">100×"]
    ax.bar(range(len(rr)), rr.values * 100, color=ORANGE, width=0.62)
    ax.set_xticks(range(len(rr)), plabels[:len(rr)])
    ax.set_xlabel("Population ratio between the two cities", color=INK2, fontsize=9)
    style(ax)
    ax.set_title("Unequal pairs rarely reciprocate", loc="left", family="Georgia", fontsize=11.5, color=INK)
    fig.tight_layout()
    save(fig, "reciprocity.png")


def distance_reach(nodes, e, geo):
    wq = lambda x, w, q: float(np.interp(q, np.cumsum(w[np.argsort(x)]) / w.sum(), np.sort(x)))
    d, w = e.dist.to_numpy(), e.raw_weight.to_numpy(float)
    S["dist_w_median"] = wq(d, w, 0.5)
    S["dist_w_p90"] = wq(d, w, 0.9)
    S["share_w_lt10"] = float(w[d < 10].sum() / w.sum())
    S["share_w_lt25"] = float(w[d < 25].sum() / w.sum())
    S["share_w_gt200"] = float(w[d > 200].sum() / w.sum())
    by_cls = e.groupby("class_t", observed=True).apply(lambda x: wq(x.dist.to_numpy(), x.raw_weight.to_numpy(float), 0.5))
    T["dist_by_target_class"] = {str(k): float(v) for k, v in by_cls.items()}
    # reach of each city: median distance its citations come from
    g = e.groupby("target")
    reach = g.apply(lambda x: wq(x.dist.to_numpy(), x.raw_weight.to_numpy(float), 0.5)).rename("reach_km")
    nodes = nodes.join(reach, on="insee")
    big = nodes[nodes.in_degree >= 60]
    cols = ["city", "department_name", "population", "in_strength", "in_degree", "reach_km"]
    T["far_reach"] = big.nlargest(15, "reach_km")[cols].to_dict("records")
    T["local_reach"] = nodes[nodes.population >= 100_000].nsmallest(10, "reach_km")[cols].to_dict("records")

    # map: who is cited from far away
    fig = plt.figure(figsize=(8, 8.6))
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.84])
    geo["outline"].plot(ax=ax, facecolor="#f3eee5", edgecolor="none", zorder=0)
    geo["reg"].boundary.plot(ax=ax, color=A.REG_LINE, linewidth=0.5, zorder=1)
    geo["outline"].boundary.plot(ax=ax, color=A.OUTLINE, linewidth=0.6, zorder=1)
    s = nodes[nodes.in_degree >= 10].sort_values("in_strength")
    sz = 2 + 260 * (np.sqrt(s.in_strength) / np.sqrt(s.in_strength.max())) ** 1.3
    sc = ax.scatter(s.x, s.y, s=sz, c=np.log10(s.reach_km.clip(3, 600)), cmap=SEQ, vmin=np.log10(3), vmax=np.log10(600),
                    linewidths=0.3, edgecolors=BG, alpha=0.9, zorder=3)
    texts = []
    for r in pd.concat([s.nlargest(14, "in_strength"), big.nlargest(8, "reach_km")]).drop_duplicates("insee").itertuples():
        t = ax.text(r.x, r.y, r.city, fontsize=7.2, color=INK, zorder=5)
        halo(t)
        texts.append(t)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=INK3, lw=0.4), iter_lim=300)
    ax.set_aspect("equal")
    ax.axis("off")
    cax = fig.add_axes([0.08, 0.045, 0.4, 0.014])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    ticks = [3, 10, 30, 100, 300]
    cb.set_ticks(np.log10(ticks), labels=[str(t) for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.set_label("median distance of the citing cities (km, citation-weighted)", fontsize=8, color=INK2)
    fig.text(0.05, 0.965, "Reach: from how far away is a city cited?", fontsize=17, family="Georgia", color=INK,
             va="top")
    fig.text(0.05, 0.93, "Cities cited by at least 10 others. Size = mentions received; darker = cited from farther away.",
             fontsize=8.8, color=INK2, va="top")
    save(fig, "reach_map.png")
    return nodes


def sections(m):
    W = m.mentions.sum()
    grp = m.groupby("group").agg(mentions=("mentions", "sum"), links=("n_links", "sum"))
    grp["share"] = grp.mentions / W
    wq = lambda x: float(np.interp(0.5, np.cumsum(x.mentions.to_numpy()[np.argsort(x.dist.to_numpy())]) /
                                   x.mentions.sum(), np.sort(x.dist.to_numpy())))
    grp["median_km"] = m.groupby("group").apply(wq)
    grp["share_paris"] = m[m.target == "75056"].groupby("group").mentions.sum() / grp.mentions
    grp["share_gt100"] = m[m.dist > 100].groupby("group").mentions.sum() / grp.mentions
    grp = grp.fillna(0).sort_values("share", ascending=True)
    T["sections"] = grp.reset_index().to_dict("records")

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    yy = np.arange(len(grp))
    for ax, col, lab, color, fmt in ((axes[0], "share", "% of all mentions", BLUE, "{:.0%}"),
                                     (axes[1], "median_km", "median distance (km)", ORANGE, "{:.0f}"),
                                     (axes[2], "share_gt100", "% of mentions to cities >100 km away", "#4a3aa7", "{:.0%}")):
        vals = grp[col].to_numpy() * (100 if "%" in lab else 1)
        ax.barh(yy, vals, color=color, height=0.62)
        for y_, v, raw in zip(yy, vals, grp[col]):
            ax.text(v + vals.max() * 0.02, y_, fmt.format(raw), va="center", fontsize=7.5, color=INK2)
        ax.set_xlabel(lab, color=INK2, fontsize=8.8)
        ax.set_xlim(0, vals.max() * 1.22)
        style(ax, "x")
    axes[0].set_yticks(yy, grp.index)
    fig.suptitle("Where in the article do cities cite each other?", x=0.012, ha="left", family="Georgia",
                 fontsize=13, color=INK)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.15, top=0.9, wspace=0.12)
    save(fig, "sections.png")


def paris(nodes, e, m, geo):
    P = "75056"
    citing = set(e.loc[e.target == P, "source"])
    nodes["cites_paris"] = nodes.insee.isin(citing)
    S["n_cite_paris"] = len(citing)
    S["pct_cite_paris"] = len(citing) / len(nodes)
    pos = nodes.set_index("insee")
    px, py = pos.loc[P, "x"], pos.loc[P, "y"]
    nodes["d_paris"] = np.hypot(nodes.x - px, nodes.y - py) / 1000
    bands = [0, 25, 50, 100, 200, 400, 1200]
    pb = nodes[nodes.insee != P].groupby(pd.cut(nodes.d_paris, bands, right=False), observed=True)["cites_paris"].mean()
    T["paris_by_distance"] = {str(k): float(v) for k, v in pb.items()}
    pm = m[m.target == P].groupby("group").mentions.sum().sort_values(ascending=False)
    T["paris_sections"] = (pm / pm.sum()).to_dict()
    # benchmark: other big cities, share of communes 100-200 km away that cite them
    bench = []
    for code in nodes.nlargest(12, "population").insee:
        cx, cy = pos.loc[code, "x"], pos.loc[code, "y"]
        dd = np.hypot(nodes.x - cx, nodes.y - cy) / 1000
        cit = nodes.insee.isin(set(e.loc[e.target == code, "source"]))
        bench.append(dict(city=pos.loc[code, "city"], pct_lt50=float(cit[(dd < 50) & (dd > 0)].mean()),
                          pct_100_200=float(cit[(dd >= 100) & (dd < 200)].mean()),
                          pct_gt300=float(cit[dd >= 300].mean()), citing=int(cit.sum())))
    T["big_city_pull"] = bench

    # choropleth: share of each département's communes citing Paris
    dep = geo["dep"].copy()
    share = nodes.groupby("department")["cites_paris"].mean()
    dep["share"] = dep["code"].map(share)
    fig = plt.figure(figsize=(7.4, 8))
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.82])
    dep.plot(ax=ax, column="share", cmap=SEQ, vmin=0, vmax=0.5, edgecolor=BG, linewidth=0.5)
    geo["outline"].boundary.plot(ax=ax, color=A.OUTLINE, linewidth=0.6)
    for r in dep[~dep.code.isin(["75", "92", "93", "94"])].nlargest(5, "share").itertuples():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, f"{r.share:.0%}", fontsize=7, ha="center", va="center", color="white", fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    cax = fig.add_axes([0.08, 0.05, 0.4, 0.014])
    sm_ = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 0.5))
    cb = fig.colorbar(sm_, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], labels=["0%", "10%", "20%", "30%", "40%", "50%+"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.set_label("share of the département's communes whose article mentions Paris", fontsize=8, color=INK2)
    fig.text(0.05, 0.965, "Paris's pull", fontsize=17, family="Georgia", color=INK, va="top")
    fig.text(0.05, 0.93, f"{len(citing):,} of {len(nodes):,} commune articles ({len(citing) / len(nodes):.0%}) mention "
             "Paris at least once in their body text.", fontsize=8.8, color=INK2, va="top")
    save(fig, "paris_pull.png")


def hubs_bridges(nodes, e, memb, g):
    W = e.raw_weight.sum()
    intra = e.comm_s == e.comm_t
    S["share_intra_community"] = float(e.loc[intra, "raw_weight"].sum() / W)
    S["n_communities"] = int(len(np.unique(memb)))
    rows = []
    ci = e[intra].groupby(["comm_t", "target"]).raw_weight.sum()
    name = nodes.set_index("insee")["city"]
    for c in np.unique(memb):
        if c not in ci.index.get_level_values(0):
            continue
        s = ci.loc[c].sort_values(ascending=False)
        size = int((nodes.community == c).sum())
        rows.append(dict(community=int(c), hub=name[s.index[0]], size=size, hub_share=float(s.iloc[0] / s.sum()),
                         top3_share=float(s.iloc[:3].sum() / s.sum()), second=name[s.index[1]],
                         hub_pop=float(nodes.set_index("insee").loc[s.index[0], "population"])))
    hubs = pd.DataFrame(rows).sort_values("hub_share", ascending=False)
    T["hubs"] = hubs.to_dict("records")
    S["hub_share_median"] = float(hubs.hub_share.median())
    S["hub_is_largest"] = int(sum(
        nodes[nodes.community == r.community].nlargest(1, "population").city.iloc[0] == r.hub for r in hubs.itertuples()))

    # bridges: participation coefficient over undirected weighted ties
    und = pd.concat([e[["source", "target", "raw_weight", "comm_t"]].rename(columns={"source": "a", "target": "b", "comm_t": "cb"}),
                     e[["target", "source", "raw_weight", "comm_s"]].rename(columns={"target": "a", "source": "b", "comm_s": "cb"})])
    k = und.groupby("a").raw_weight.sum()
    kc = und.groupby(["a", "cb"]).raw_weight.sum()
    P = 1 - (kc / kc.index.get_level_values(0).map(k).to_numpy()).pow(2).groupby(level=0).sum()
    nodes = nodes.join(P.rename("participation"), on="insee")
    br = nodes[(nodes.strength >= 150)].nlargest(15, "participation")
    T["bridges"] = br[["city", "department_name", "population", "strength", "participation"]].to_dict("records")
    return nodes


def templates(nodes, e, m):
    """'Template stars': ordinary communes cited by many articles, almost only in Géographie, mostly
    from one département -> the signature of a sentence copied across a département's articles."""
    info = nodes.set_index("insee")
    tgt = m.groupby("target").agg(mentions=("mentions", "sum"))
    tgt["geo_share"] = m[m.group == "Geography"].groupby("target").mentions.sum() / tgt.mentions
    src_dep = e.assign(dep=e.source.map(info["department"]))
    modal = src_dep.groupby(["target", "dep"]).size().groupby(level=0).max()
    tgt["in_degree"] = e.groupby("target").size()
    tgt["modal_dep_share"] = modal / tgt.in_degree
    tgt = tgt.join(info[["city", "department_name", "population", "seat"]])
    stars = tgt[(tgt.in_degree >= 80) & (tgt.geo_share >= 0.85) & (tgt.modal_dep_share >= 0.7)
                & (tgt.seat == "Ordinary commune")].sort_values("in_degree", ascending=False)
    T["template_stars"] = stars.reset_index()[["city", "department_name", "population", "in_degree", "geo_share",
                                               "modal_dep_share"]].head(15).to_dict("records")
    S["n_template_stars"] = len(stars)
    S["template_star_ties"] = int(stars.in_degree.sum())
    S["template_star_share_ties"] = float(stars.in_degree.sum() / len(e))
    # same test for administrative seats: how much of their in-degree is Géographie-only text
    seats = tgt[tgt.seat != "Ordinary commune"]
    S["seat_geo_share_median"] = float(seats.geo_share.median())
    S["ordinary_geo_share_median"] = float(tgt[tgt.seat == "Ordinary commune"].geo_share.median())


def strongest(nodes, e):
    name = nodes.set_index("insee")["city"]
    top = e.nlargest(15, "raw_weight")
    T["top_ties"] = [dict(a=name[r.source], b=name[r.target], w=int(r.raw_weight), links=int(r.n_links),
                          dist=float(r.dist)) for r in top.itertuples()]
    far = e[(e.dist > 300) & (e.target != "75056") & (e.source != "75056")].nlargest(12, "raw_weight")
    T["long_ties"] = [dict(a=name[r.source], b=name[r.target], w=int(r.raw_weight), dist=float(r.dist))
                      for r in far.itertuples()]
    # targets of long-range citations
    lr = e[e.dist > 300].groupby("target").raw_weight.sum().sort_values(ascending=False)
    S["share_longrange_top10"] = float(lr.head(10).sum() / lr.sum())
    T["longrange_targets"] = [dict(city=name[c], w=int(v), share=float(v / lr.sum())) for c, v in lr.head(12).items()]


def main():
    data, nodes, e, u, m, g, memb, gamma = load()
    geo = A.prepare_geodata(data)
    overview(nodes, e, m)
    who_cites(nodes)
    who_is_cited(nodes, e)
    hierarchy(nodes, e)
    reciprocity(nodes, e)
    nodes = distance_reach(nodes, e, geo)
    sections(m)
    paris(nodes, e, m, geo)
    nodes = hubs_bridges(nodes, e, memb, g)
    strongest(nodes, e)
    templates(nodes, e, m)
    S["gamma"] = gamma
    # reuse figures from the main pipeline
    for f in ("france_city_citation_network_linkedin.png", "citation_communities_vs_regions.png", "distance_decay.png",
              "citation_territories_extended.png"):
        shutil.copy(A.OUT / f, FIGS / f)
    comp = pd.read_csv(A.OUT / "community_admin_comparison.csv")
    S["admin"] = comp[comp.primary].iloc[0][[c for c in comp.columns if c.startswith(("ARI_", "NMI_"))]].to_dict()
    perm = pd.read_csv(A.OUT / "permutation_tests.csv")
    T["perm"] = perm.to_dict("records")
    (REPORT / "report_numbers.json").write_text(
        json.dumps(dict(S=S, T=T), ensure_ascii=False, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)),
        encoding="utf-8")
    print(json.dumps(S, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
