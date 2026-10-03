"""Validate report assets, input fingerprints, and conservation in exported networks."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value and not value.startswith(("http", "#")):
                self.links.append(value)


def main():
    content = json.loads((REPORT / "section_report.json").read_text(encoding="utf-8"))
    for name, expected in content["provenance"]["input_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"Changed input: {name}"
    parser = Links()
    parser.feed((REPORT / "index.html").read_text(encoding="utf-8"))
    for name in parser.links:
        assert (REPORT / name).is_file(), f"Missing report asset: {name}"
    assert len(content["layers"]) == 5
    summary = pd.read_csv(REPORT / "section_summary.csv").set_index("key")
    for layer in content["layers"]:
        key = layer["key"]
        nodes = pd.read_csv(REPORT / "maps" / f"{key}_nodes.csv", dtype={"insee": str})
        edges = pd.read_csv(REPORT / "maps" / f"{key}_edges.csv.gz", dtype={"source": str, "target": str})
        assert nodes.insee.is_unique
        assert not edges.duplicated(["source", "target"]).any()
        assert edges.source.ne(edges.target).all()
        assert set(edges.source) | set(edges.target) <= set(nodes.insee)
        assert edges.raw_weight.gt(0).all()
        assert edges.raw_weight.eq(edges.n_links + edges.n_text).all()
        assert edges.raw_weight.sum() == nodes.in_strength.sum() == nodes.out_strength.sum() == layer["mentions"]
        assert len(edges) == layer["directed_edges"] == summary.loc[key, "directed_edges"]
        np.testing.assert_allclose(edges.groupby("source").p_source_norm.sum(), 1.0)
        assert nodes.community.notna().sum() == layer["lcc_nodes"]
        assert nodes.community.nunique() == layer["communities"]
        assert nodes.strength.gt(0).sum() == layer["active_nodes"]
        assert layer["visual_edges"] <= layer["n_edges_lcc"] <= layer["directed_edges"]
        assert np.isfinite(edges.distance_km).all()
        for extension in ("png", "jpg"):
            with Image.open(REPORT / "maps" / f"{key}.{extension}") as image:
                assert image.size == (2400, 3000)
                image.verify()
        print(f"{key}: counts, source weights, cluster coverage and maps OK")
    assert (REPORT / "citation_report.pdf").read_bytes().startswith(b"%PDF-")
    territory = json.loads((REPORT / "maps" / "all_territories.json").read_text())
    all_nodes = pd.read_csv(REPORT / "maps" / territory["nodes_file"], dtype={"insee": str})
    clustered = all_nodes[all_nodes.community.notna()].sort_values("insee")
    membership = [(str(r.insee), int(r.community)) for r in clustered.itertuples()]
    assert hashlib.sha256(json.dumps(membership).encode("utf-8")).hexdigest() == territory["membership_sha256"]
    all_layer = next(s for s in content["layers"] if s["key"] == "all")
    assert territory["gamma"] == all_layer["gamma"]
    assert territory["communities"] == all_layer["communities"]
    with Image.open(REPORT / "maps" / "all_territories.png") as image:
        assert image.size == (2400, 3000)
        image.verify()
    print("Territories reuse the exact overall-map partition.")
    print("Report links, input hashes, five exported networks and PDF validated.")


if __name__ == "__main__":
    main()
