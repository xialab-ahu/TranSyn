"""
@description: test context-overlap extent calculations
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import unittest

import pandas as pd

from calculate_context_overlap import KEYS
from calculate_overlap_extent import measure_overlap, summarize


class OverlapExtentTests(unittest.TestCase):
    """Validate coordinates and denominators with boundary examples."""

    def test_overlap_boundaries_and_missing_chromosome(self):
        """
        Desc:
            Execute the test overlap boundaries and missing chromosome operation.
        """
        source = pd.DataFrame(
            [("1", p, "A", "G") for p in [1000, 1001, 1064, 1128, 1129]] + [("2", 1000, "A", "G")],
            columns=KEYS,
        )
        target = pd.DataFrame([("1", 1000, "A", "T")], columns=KEYS)
        result = measure_overlap(source, target)
        self.assertEqual(result["max_overlap_bp"].tolist(), [129, 128, 65, 1, 0, 0])
        self.assertTrue(result.loc[result["max_overlap_bp"].eq(0), "overlap_start"].isna().all())
        self.assertTrue(result.iloc[-1]["nearest_distance_bp"] is pd.NA)

    def test_all_record_normalization_and_thresholds(self):
        """
        Desc:
            Execute the test all record normalization and thresholds operation.
        """
        source = pd.DataFrame([("1", 1000, "A", "G"), ("2", 1000, "A", "G")], columns=KEYS)
        target = pd.DataFrame([("1", 1000, "A", "T")], columns=KEYS)
        detail = measure_overlap(source, target)
        summary, dist, thresholds, curve = summarize(detail, "combined", "missense", "variants")
        self.assertEqual(summary["normalized_overlap_percent_all"], 50)
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(dist["count"].sum(), 2)
        self.assertEqual(thresholds["minimum_overlap_bp"].tolist(), [1, 33, 65, 97, 117, 129])
        self.assertTrue(thresholds["percent_all"].eq(50).all())
        self.assertEqual(curve.iloc[0]["percent_all"], 100)

    def test_reverse_direction_uses_its_own_population(self):
        """
        Desc:
            Execute the test reverse direction uses its own population operation.
        """
        source = pd.DataFrame([("1", 1000, "A", "G"), ("2", 1000, "A", "G")], columns=KEYS)
        synonymous = pd.DataFrame([("1", 1000, "A", "T")], columns=KEYS)
        forward = measure_overlap(source, synonymous)
        reverse = measure_overlap(synonymous, source)
        self.assertEqual(forward["max_overlap_fraction"].mean(), 0.5)
        self.assertEqual(reverse["max_overlap_fraction"].mean(), 1.0)

    def test_nearest_single_window_and_ties(self):
        """
        Desc:
            Execute the test nearest single window and ties operation.
        """
        source = pd.DataFrame([("1", 1000, "A", "G")], columns=KEYS)
        target = pd.DataFrame([("1", p, "A", "T") for p in [990, 1010, 1200]], columns=KEYS)
        result = measure_overlap(source, target)
        self.assertEqual(result.iloc[0]["max_overlap_bp"], 119)
        self.assertEqual(result.iloc[0]["reference_pos"], 990)
        self.assertEqual(result.iloc[0]["overlap_start"], 936)
        self.assertEqual(result.iloc[0]["overlap_end"], 1054)


if __name__ == "__main__":
    unittest.main()
