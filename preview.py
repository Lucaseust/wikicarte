"""Fast figure iteration using a verified cached Leiden partition."""
import numpy as np
import argparse
import pandas as pd
import shutil
import analysis as A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-output", action="store_true", help="Refresh the four extended network maps from verified clusters")
    args = parser.parse_args()
    data = A.load_data()
    nodes, s2i, t2i = A.clean_city_identifiers(data)
    edges = A.build_network(data["mentions"], nodes, s2i, t2i)
    e, u, nodes = A.calculate_edge_weights(edges, nodes)
    g, keep = A.lcc_graph(u, nodes)
    sol, parts, _ = A.load_partition_cache(g)
    gamma = A.select_primary(sol)
    memb = parts[(A.PRIMARY_WEIGHTING, gamma)]
    nd = nodes.set_index("insee").loc[g.vs["insee"]].reset_index()
    names = {c: nd[memb == c].nlargest(1, "in_strength")["city"].iloc[0] for c in np.unique(memb)}
    sizes = np.bincount(memb)
    major = [c for c in np.argsort(-sizes) if sizes[c] >= 50]
    colours = A.adjacency_colouring(memb, nd[["x", "y"]].to_numpy() / 1000, major)
    geo = A.prepare_geodata(data)
    meta = dict(gamma=gamma, n_nodes=len(nodes), W=int(e.raw_weight.sum()), n_edges=len(e),
                n_edges_lcc=int((e.source.isin(set(keep)) & e.target.isin(set(keep))).sum()))
    A.plot_linkedin_map(geo, nodes, g, memb, e, colours, {c: names[c] for c in major}, meta,
                        A.HERE / "cache" / "preview_linkedin.png")
    if args.refresh_output:
        main_png = A.OUT / "france_city_citation_network_linkedin.png"
        A.plot_linkedin_map(geo, nodes, g, memb, e, colours, {c: names[c] for c in major}, meta,
                            main_png, A.OUT / "france_city_citation_network_linkedin.svg")
        A.plot_linkedin_map(geo, nodes, g, memb, e, colours, {c: names[c] for c in major}, dict(meta),
                            A.OUT / "france_city_citation_network_full.png", full=True)
        comp = pd.read_csv(A.OUT / "community_admin_comparison.csv")
        A.plot_admin_comparison(geo, nodes, g, memb, colours, names, comp[comp.primary].iloc[0],
                                A.OUT / "citation_communities_vs_regions.png")
        A.plot_territories(geo, nodes, g, memb, colours, names, A.OUT / "citation_territories_extended.png",
                           meta=dict(gamma=gamma, title="Citation territories — extended analysis"))
        for filename in (main_png.name, "france_city_citation_network_full.png",
                         "citation_communities_vs_regions.png", "citation_territories_extended.png"):
            shutil.copy(A.OUT / filename, A.HERE / "report" / "figs" / filename)


if __name__ == "__main__":
    main()
