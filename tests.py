import unittest
import numpy as np

from taxonomy import Taxonomy


class TestTaxonomy(unittest.TestCase):
    def test_make_taxonomy(self):
        """
        Test that it can make a taxonomy."
        """
        tax = Taxonomy("fixtures/taxonomy_fixture.csv")
        self.assertIsNotNone(tax)

    def test_taxonomy_grafts_life(self):
        """
        Test that the taxonomy grafts the life node and that
        the life node has 5 children.
        """
        tax = Taxonomy("fixtures/taxonomy_fixture.csv")
        self.assertIs(len(tax.life.children), 5)

    def test_taxonomy_human_branch(self):
        """
        Test that the taxonomy has a human node, and that the
        human branch has the expected taxonomy. Note this test
        may fail if the provided taxonomy drifts.
        """
        tax = Taxonomy("fixtures/taxonomy_fixture.csv")
        human_node = tax.nodes_by_taxon_id[43584]
        self.assertIsNotNone(human_node)

        human_ancestry = []

        node_id = human_node.taxon_id
        node = human_node
        while node is not tax.life:
            human_ancestry.append(node_id)

            node_id = node.parent
            node = tax.nodes_by_taxon_id[node_id]
        human_ancestry = list(reversed(human_ancestry))

        self.assertListEqual(
            human_ancestry,
            [
                1,
                2,
                355675,
                40151,
                848317,
                848320,
                848323,
                43367,
                786045,
                936377,
                936369,
                1036675,
                43575,
                846252,
                43583,
                43584,
            ],
        )

    def test_taxonomy_leaf_node_count(self):
        tax = Taxonomy("fixtures/taxonomy_fixture.csv")
        self.assertEqual(len(tax.nodes_by_leaf_class_id.keys()), 502)

    def test_taxonomy_agg_sum_to_one(self):
        tax = Taxonomy("fixtures/taxonomy_fixture.csv")
        flat_scores = [1 / 502] * 502
        agg_scores = tax.aggregate_scores(flat_scores, tax.life)
        self.assertAlmostEqual(agg_scores[tax.life.taxon_id], 1.0)

    def test_taxonomy_flat_best_branch(self):
        tax = Taxonomy("fixtures/taxonomy_fixture.csv")
        flat_scores = [1 / 502] * 502
        agg_scores = tax.aggregate_scores(flat_scores, tax.life)
        best_branch = tax.best_branch_from_scores(agg_scores)
        bb_taxon_ids = [bb_node.taxon_id for bb_node, score in best_branch]
        self.assertListEqual(
            bb_taxon_ids,
            [
                48460,
                1,
                47120,
                372739,
                47158,
                184884,
                47157,
                47224,
                47922,
                202067,
                202066,
                50392,
                54064,
            ],
        )


if __name__ == "__main__":
    unittest.main()
