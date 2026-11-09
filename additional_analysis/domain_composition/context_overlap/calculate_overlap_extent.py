"""
@description: quantify the maximum overlap between variant contexts
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from calculate_context_overlap import BASE, GROUPS, KEYS, load_variants, sha256

CONTEXT_BP = 129
FLANK_BP = 64


def measure_overlap(source, reference_variants):
    """
    Desc:
        Execute the measure overlap operation.
    Args:
        source, Source variant collection.
        reference_variants, Value supplied for reference_variants.
    """
    source = source[KEYS].drop_duplicates().copy()
    reference = (
        reference_variants[["chrom", "pos"]]
        .drop_duplicates()
        .rename(columns={"pos": "reference_pos"})
    )

    detail = pd.merge_asof(
        source.sort_values("pos"),
        reference.sort_values("reference_pos"),
        left_on="pos",
        right_on="reference_pos",
        by="chrom",
        direction="nearest",
    )
    detail["reference_pos"] = detail["reference_pos"].astype("Int64")
    detail["nearest_distance_bp"] = (detail["pos"] - detail["reference_pos"]).abs()

    detail["max_overlap_bp"] = (
        (CONTEXT_BP - detail["nearest_distance_bp"]).clip(lower=0).fillna(0).astype(int)
    )
    detail["max_overlap_fraction"] = detail["max_overlap_bp"] / CONTEXT_BP
    detail["max_overlap_percent"] = 100 * detail["max_overlap_fraction"]
    detail["context_start"] = detail["pos"] - FLANK_BP
    detail["context_end"] = detail["pos"] + FLANK_BP
    detail["overlap_start"] = (
        detail[["pos", "reference_pos"]].max(axis=1).astype("Int64") - FLANK_BP
    )
    detail["overlap_end"] = detail[["pos", "reference_pos"]].min(axis=1).astype("Int64") + FLANK_BP

    detail.loc[detail["max_overlap_bp"].eq(0), ["overlap_start", "overlap_end"]] = pd.NA
    detail["variant38"] = (
        detail["chrom"]
        + "_"
        + detail["pos"].astype(str)
        + "_"
        + detail["ref"]
        + "/"
        + detail["alt"]
    )
    return detail.sort_values(KEYS).reset_index(drop=True)


def summarize(detail, group, dataset, unit):
    """
    Desc:
        Execute the summarize operation.
    Args:
        detail, Value supplied for detail.
        group, Variant-label group.
        dataset, Value supplied for dataset.
        unit, Value supplied for unit.
    """
    frame = detail if unit == "variants" else detail.drop_duplicates(["chrom", "pos"])
    length = frame["max_overlap_bp"]
    total = len(frame)
    if not total:
        raise ValueError(f"{group}/{dataset}/{unit}: no records; normalized overlap is undefined")
    identity = {"group": group, "dataset": dataset, "unit": unit}

    summary = {
        **identity,
        "total_records": total,
        "zero_overlap_records": int(length.eq(0).sum()),
        "overlapping_records": int(length.gt(0).sum()),
        "any_overlap_percent": 100 * length.gt(0).mean(),
        "sum_max_overlap_bp": int(length.sum()),
        "normalization_denominator_bp": CONTEXT_BP * total,
        "mean_max_overlap_bp_all": length.mean(),
        "normalized_overlap_fraction_all": length.sum() / (CONTEXT_BP * total),
        "normalized_overlap_percent_all": 100 * length.sum() / (CONTEXT_BP * total),
        "median_max_overlap_bp_all": length.median(),
        "p90_max_overlap_bp_all": length.quantile(0.90),
        "p95_max_overlap_bp_all": length.quantile(0.95),
        "p99_max_overlap_bp_all": length.quantile(0.99),
    }

    counts = length.value_counts().reindex(range(CONTEXT_BP + 1), fill_value=0)
    distribution = counts.rename_axis("max_overlap_bp").reset_index(name="count")
    distribution = distribution.assign(**identity, total_records=total)
    distribution["max_overlap_fraction"] = distribution["max_overlap_bp"] / CONTEXT_BP
    distribution["fraction_all"] = distribution["count"] / total
    distribution["percent_all"] = 100 * distribution["fraction_all"]
    curve = distribution[["group", "dataset", "unit", "total_records"]].copy()
    curve["minimum_overlap_bp"] = distribution["max_overlap_bp"]
    curve["minimum_overlap_fraction"] = curve["minimum_overlap_bp"] / CONTEXT_BP
    curve["records_at_or_above"] = counts.iloc[::-1].cumsum().iloc[::-1].to_numpy()
    curve["fraction_all"] = curve["records_at_or_above"] / total
    curve["percent_all"] = 100 * curve["fraction_all"]
    thresholds = []

    for label, fraction in [
        ("any_overlap", 1 / CONTEXT_BP),
        ("25%", 0.25),
        ("50%", 0.5),
        ("75%", 0.75),
        ("90%", 0.9),
        ("100%", 1.0),
    ]:
        minimum = int(np.ceil(fraction * CONTEXT_BP))
        count = int(length.ge(minimum).sum())
        thresholds.append(
            {
                **identity,
                "threshold": label,
                "minimum_overlap_fraction": fraction,
                "minimum_overlap_bp": minimum,
                "records_at_or_above": count,
                "total_records": total,
                "fraction_all": count / total,
                "percent_all": 100 * count / total,
            }
        )
    assert counts.sum() == total
    assert np.isclose(distribution["fraction_all"].sum(), 1)
    assert np.isclose(
        (distribution["max_overlap_fraction"] * distribution["fraction_all"]).sum(),
        summary["normalized_overlap_fraction_all"],
    )
    return summary, distribution, pd.DataFrame(thresholds), curve


def main():
    """
    Desc:
        Run the command-line workflow.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--denominator",
        choices=("source", "synonymous"),
        default="source",
        help="source uses noncoding/missense as denominator; synonymous uses synonymous variants",
    )
    for dataset in ("missense", "synonymous", "noncoding"):
        for group in ("benign", "pathogenic"):
            parser.add_argument(f"--{dataset}-{group}", type=Path, required=True)
    args = parser.parse_args()
    input_paths = {
        dataset: {group: getattr(args, f"{dataset}_{group}") for group in ("benign", "pathogenic")}
        for dataset in ("missense", "synonymous", "noncoding")
    }
    default_name = (
        "statistics_extent" if args.denominator == "source" else "statistics_extent_synonymous"
    )
    output = (args.output_dir or BASE / default_name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output / "statistics.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info(
        "Python: %s; pandas: %s; NumPy: %s", sys.executable, pd.__version__, np.__version__
    )
    data, qc, inputs = {}, [], []
    for dataset, paths in input_paths.items():
        for group, path in paths.items():
            data[(dataset, group)], quality = load_variants(path)
            qc.append({"dataset": dataset, "group": group, **quality})
            inputs.append(
                {"dataset": dataset, "group": group, "path": str(path), "sha256": sha256(path)}
            )
        data[(dataset, "combined")] = pd.concat(
            [data[(dataset, "benign")], data[(dataset, "pathogenic")]], ignore_index=True
        ).drop_duplicates(KEYS)
    summaries, distributions, thresholds, curves, details = [], [], [], [], []
    for group in GROUPS:
        for dataset in ("noncoding", "missense"):
            denominator = dataset if args.denominator == "source" else "synonymous"
            reference = "synonymous" if args.denominator == "source" else dataset

            detail = measure_overlap(data[(denominator, group)], data[(reference, group)])
            detail = detail.assign(
                group=group,
                dataset=dataset,
                denominator_dataset=denominator,
                reference_dataset=reference,
            )
            details.append(detail)
            for unit in ("variants", "sites"):
                summary, distribution, threshold, curve = summarize(detail, group, dataset, unit)
                summary.update(denominator_dataset=denominator, reference_dataset=reference)
                for frame in (distribution, threshold, curve):
                    frame["denominator_dataset"] = denominator
                    frame["reference_dataset"] = reference
                summaries.append(summary)
                distributions.append(distribution)
                thresholds.append(threshold)
                curves.append(curve)
                logging.info(
                    "%s/%s/%s: N=%d; normalized overlap=%.6f%%; any overlap=%.6f%%",
                    group,
                    dataset,
                    unit,
                    summary["total_records"],
                    summary["normalized_overlap_percent_all"],
                    summary["any_overlap_percent"],
                )
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "summary_all_units.tsv", sep="\t", index=False)

    plotting = summary_table.loc[summary_table["unit"].eq("variants")].copy()
    plotting = plotting.rename(
        columns={
            "dataset": "source",
            "total_records": "source_unique_variants",
            "overlapping_records": "overlapping_source_variants",
            "normalized_overlap_percent_all": "overlap_percent",
        }
    )
    sites = summary_table.loc[
        summary_table["unit"].eq("sites"),
        [
            "group",
            "dataset",
            "total_records",
            "overlapping_records",
            "normalized_overlap_percent_all",
        ],
    ].rename(
        columns={
            "dataset": "source",
            "total_records": "source_unique_sites",
            "overlapping_records": "overlapping_source_sites",
            "normalized_overlap_percent_all": "site_overlap_percent",
        }
    )
    plotting = plotting.merge(sites, on=["group", "source"], validate="one_to_one")
    plotting = plotting.assign(target="synonymous", metric="mean_max_overlap_fraction_all")
    if args.denominator == "synonymous":
        plotting["target"] = plotting["source"]
        plotting["source"] = "synonymous"
    plotting.to_csv(output / "overlap_with_synonymous.tsv", sep="\t", index=False)

    summary_table.loc[summary_table["unit"].eq("variants")].to_csv(
        output / "summary_variants.tsv", sep="\t", index=False
    )
    for name, frames in [
        ("per_variant_overlap", details),
        ("overlap_distribution", distributions),
        ("overlap_thresholds", thresholds),
        ("overlap_survival_curve", curves),
    ]:
        pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.tsv", sep="\t", index=False)
    pd.DataFrame(qc).to_csv(output / "dataset_qc.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.executable, "pandas": pd.__version__, "numpy": np.__version__},
        "context": "hg38 input coordinates; 1-based inclusive [POS-64, POS+64]; strand ignored",
        "score": "max(0,129 - nearest reference center distance on the same chromosome) / 129",
        "denominator_side": args.denominator,
        "normalization": "sum(max_overlap_bp) / (129 * all unique source variants); zero overlaps included",
        "denominators": "Each dataset and label group independently; not pooled across missense/noncoding",
        "combined": "Deduplicated union of benign/pathogenic within each dataset; nearest neighbors recomputed",
        "matching": "Single best reference context per denominator variant; no union of multiple windows; ties pick lower position",
        "sequence_columns": "Not read or validated",
        "chromosome_ends": "No chromosome-end clipping; uses nominal 129 bp windows",
        "inputs": inputs,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    logging.info("Analysis complete: %s", output)


if __name__ == "__main__":
    main()
