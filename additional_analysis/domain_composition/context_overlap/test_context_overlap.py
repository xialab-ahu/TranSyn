"""
@description: test context-overlap calculations
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import unittest

from calculate_context_overlap import (
    covered_bp,
    directional_stats,
    index_sites,
    intersection_bp,
    merged_intervals,
    neighbor_count,
)


class ContextOverlapTests(unittest.TestCase):
    """Validate the core statistics with manually verifiable examples."""

    def test_inclusive_boundary_and_chromosomes(self):
        """
        Desc:
            Execute the test inclusive boundary and chromosomes operation.
        """
        self.assertEqual(neighbor_count([228, 229], 100, 64), 1)
        self.assertEqual(neighbor_count([100], 228, 64), 1)
        self.assertEqual(neighbor_count([100], 229, 64), 0)
        self.assertNotIn("2", index_sites({("1", 100, "A", "G")}))

    def test_union_coverage_and_one_base_intersection(self):
        """
        Desc:
            Execute the test union coverage and one base intersection operation.
        """
        left = merged_intervals({("1", 100, "A", "G"), ("1", 101, "C", "T")}, 64)
        right = merged_intervals({("1", 229, "A", "G"), ("2", 100, "A", "G")}, 64)
        self.assertEqual(covered_bp(left), 130)
        self.assertEqual(covered_bp(right), 258)
        self.assertEqual(intersection_bp(left, right), 1)

    def test_variant_and_site_denominators(self):
        """
        Desc:
            Execute the test variant and site denominators operation.
        """
        source = {("1", 100, "A", "G"), ("1", 100, "A", "T"), ("2", 100, "A", "G")}
        target = {("1", 228, "C", "T")}
        stats, detail = directional_stats(source, target, "combined", "a", "b", 64)
        self.assertEqual(int(detail["overlaps"].sum()), 2)
        self.assertEqual(stats["overlapping_source_variants"], 2)
        self.assertAlmostEqual(stats["overlap_percent"], 200 / 3)
        self.assertEqual(stats["site_overlap_percent"], 50)
        reverse, _ = directional_stats(target, source, "combined", "b", "a", 64)
        self.assertEqual(reverse["overlap_percent"], 100)
        self.assertEqual(
            stats["overlapping_unique_site_pairs"], reverse["overlapping_unique_site_pairs"]
        )

    def test_combined_recomputed_across_labels(self):
        """
        Desc:
            Execute the test combined recomputed across labels operation.
        """
        benign_a = {("1", 100, "A", "G")}
        pathogenic_a = {("2", 100, "A", "G")}
        benign_b = {("2", 100, "A", "T")}
        pathogenic_b = {("1", 100, "A", "T")}
        self.assertEqual(neighbor_count(index_sites(benign_b).get("1", []), 100, 64), 0)
        self.assertEqual(neighbor_count(index_sites(pathogenic_b).get("2", []), 100, 64), 0)
        combined_b = index_sites(benign_b | pathogenic_b)
        self.assertTrue(
            all(neighbor_count(combined_b[c], p, 64) for c, p, *_ in benign_a | pathogenic_a)
        )
        self.assertEqual(len(benign_a | benign_a), 1)


if __name__ == "__main__":
    unittest.main()
