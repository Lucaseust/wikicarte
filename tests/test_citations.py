"""Regression checks for extraction, weights, topic assignment and cache safety."""
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import igraph as ig
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as A  # noqa: E402
import extract_links as E  # noqa: E402
import section_maps as S  # noqa: E402
from section_groups import section_group  # noqa: E402


class ExtractionTests(unittest.TestCase):
    def test_nested_exclusions_preserve_prose(self):
        raw = '''{{Infobox|x={{nested|[[Paris]]}}}}
[[Lyon]]
<ref>[[Paris]]\n== Notes ==\nreference text</ref>
[[Bordeaux]]
[[Fichier:x.jpg|caption [[Paris]]]]
<!-- [[Paris]] --><nowiki class="example">[[Paris]]</nowiki>
{| class="wikitable"
| [[Paris]]
|}
== Voir aussi ==
[[Paris]]'''
        body = E.clean_body(raw)
        self.assertEqual([m.group(1) for m in E.LINK_RE.finditer(body)], ["Lyon", "Bordeaux"])

    def test_h2_subsection_inheritance(self):
        parts = E.split_sections("Lead\n== Histoire ==\nA\n=== Culture ===\nB\n== Culture locale ==\nC")
        self.assertEqual([s for s, _ in parts], ["lead", "histoire", "histoire", "culture locale"])
        self.assertNotIn("Culture ===", "".join(t for _, t in parts))

    def test_title_normalization(self):
        self.assertEqual(E.norm_title(" lyon_#Section "), "Lyon")
        self.assertEqual(E.norm_title("Saint-Étienne"), "Saint-Étienne")
        for title in ["Catégorie:Paris", ":Fichier:X.png", "en:Paris", "#section"]:
            self.assertIsNone(E.norm_title(title))

    def test_negative_dms(self):
        self.assertAlmostEqual(E.parse_coord("-1/30/0"), -1.5)
        self.assertAlmostEqual(E.parse_coord("1/30/0/O"), -1.5)
        self.assertAlmostEqual(E.parse_coord("48,5"), 48.5)

    def test_latest_revision_numeric_order(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            for name in ("9.wikitext.gz", "100.wikitext.gz", "notes.wikitext.gz"):
                (folder / name).touch()
            self.assertEqual(E.latest_revision(folder)[0], 100)

    def test_plain_text_link_anchor_and_apostrophe_self_name(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "Val_d_Ize"
            folder.mkdir()
            with gzip.open(folder / "100.wikitext.gz", "wt", encoding="utf-8") as fh:
                fh.write("[[Izé]] Izé\n== Histoire ==\n[[Paris]] Paris Paris Lyon")
            resolver = {t: t for t in ["Val-d'Izé", "Izé", "Paris", "Lyon"]}
            with patch.object(E, "CONTENT_DIR", Path(root)), patch.object(E, "RESOLVE", resolver), \
                    patch.object(E, "BASENAME", resolver):
                _, rows = E.process(("Val_d_Ize", "Val-d'Izé"))
            counts = {(r[1], r[2]): (r[3], r[4]) for r in rows}
            self.assertEqual(counts[("Izé", "lead")], (1, 0))
            self.assertEqual(counts[("Paris", "histoire")], (1, 2))
            self.assertFalse(any(r[1] == "Lyon" for r in rows))


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame({"insee": ["00001", "00002", "00003", "00004"]})
        self.mentions = pd.DataFrame([
            ("a", "B", "histoire", 1, 2), ("a", "B", "culture", 2, 0),
            ("b", "A", "lead", 1, 0), ("a", "A", "histoire", 3, 0),
            ("a", "unknown", "culture", 10, 0), ("a", "C", "geographie", 5, 0),
        ], columns=["source_slug", "target_title", "section", "n_links", "n_text"])
        self.data = {"mentions": self.mentions}

    def test_conservation_and_source_normalization(self):
        edges = A.build_network(self.mentions, self.nodes, {"a": "00001", "b": "00002"},
                                {"A": "00001", "B": "00002", "C": "00003"})
        e, u, nodes = A.calculate_edge_weights(edges, self.nodes)
        self.assertEqual(e.raw_weight.sum(), 11)
        self.assertEqual(u.w.sum(), 11)
        self.assertEqual(nodes.in_strength.sum(), 11)
        self.assertEqual(nodes.out_strength.sum(), 11)
        np.testing.assert_allclose(e.groupby("source").p_source_norm.sum(), 1)
        self.assertEqual(float(u.loc[u.v.eq("00002"), "source_normalized"].iloc[0]), 1.5)
        self.assertEqual(nodes.loc[nodes.insee.eq("00004"), "strength"].iloc[0], 0)

    def test_topic_subsets_conserve_counts(self):
        m = S.canonical_mentions(self.data, self.nodes, {"a": "00001", "b": "00002"},
                                 {"A": "00001", "B": "00002", "C": "00003"})
        self.assertEqual(m.mentions.sum(), 11)
        self.assertEqual(m.groupby("group").mentions.sum().sum(), 11)
        self.assertEqual(S.layer_edges(m[m.group.eq("History")]).raw_weight.sum(), 3)

    def test_small_graph_colours_and_zero_sizes(self):
        colours = A.adjacency_colouring(np.array([0, 1]), np.array([[0., 0.], [1., 0.]]), [0, 1])
        self.assertNotEqual(colours[0], colours[1])
        self.assertTrue(np.isfinite(A.node_sizes([0, 0])).all())
        self.assertEqual(len(A.node_sizes([])), 0)

    def test_empty_network_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "No positive"):
            A.lcc_graph(pd.DataFrame(), self.nodes)

    def test_fingerprint_changes_with_identifiers_and_weights(self):
        g = ig.Graph(n=2, edges=[(0, 1)])
        g.vs["insee"] = ["00001", "00002"]
        for weight in A.WEIGHTINGS:
            g.es[weight] = [1.0]
        stamp = A.graph_fingerprint(g)
        g.es["source_normalized"] = [2.0]
        self.assertNotEqual(stamp, A.graph_fingerprint(g))
        g.es["source_normalized"] = [1.0]
        g.vs["insee"] = ["00002", "00001"]
        self.assertNotEqual(stamp, A.graph_fingerprint(g))

    def test_unverified_cache_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as root, patch.object(A, "CACHE", Path(root)):
            with self.assertRaisesRegex(ValueError, "unverified or stale"):
                A.load_partition_cache(ig.Graph(n=0))

    def test_weighted_distance_median(self):
        self.assertEqual(S.weighted_median([300, 10, 50], [1, 6, 2]), 10)

    def test_normalized_topic_and_first_match(self):
        self.assertEqual(section_group("Géographie"), "Geography")
        self.assertEqual(section_group("Culture locale et patrimoine"), "Culture & heritage")
        self.assertEqual(section_group("Histoire et culture"), "History")
        self.assertEqual(section_group("Personnalités liées à la commune"), "Personalities")
        self.assertEqual(section_group("Unusual heading"), "Other")

    def test_geographic_labels_cannot_drift_to_another_region(self):
        fig, ax = A.plt.subplots()
        text = ax.text(900_000, 5_000_000, "Lyon")
        anchor = np.array([842_000., 6_519_000.])
        A.constrain_labels(ax, [text], [anchor])
        self.assertLessEqual(np.linalg.norm(np.array(text.get_position()) - anchor), 35_000.001)
        self.assertEqual(len(ax.texts), 2)  # one label plus its anchored leader annotation
        A.plt.close(fig)




if __name__ == "__main__":
    unittest.main()
