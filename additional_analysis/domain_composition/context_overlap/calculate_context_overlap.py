#!/usr/bin/env python3
"""
@description: calculate overlap between 129-bp variant contexts
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import argparse
import hashlib
import json
import logging
import sys

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GROUPS = ("benign", "pathogenic", "combined")


KEYS = ["chrom", "pos", "ref", "alt"]


def sha256(path):
    """
    Desc:
        Calculate the SHA-256 checksum of an input file.
    Args:
        path, Input filesystem path.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_variants(path):
    """
    Desc:
        Load, validate, and deduplicate an SNV table.
    Args:
        path, Input filesystem path.
    """

    coordinate_columns = {
        "chr",
        "pos_38",
        "ref",
        "alt",
        "Variant38",
        "#CHROM",
        "POS",
        "REF",
        "ALT",
        "variant38",
    }
    raw = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in coordinate_columns,
    )
    names = (
        {"chr": "chrom", "pos_38": "pos", "Variant38": "variant38"}
        if "pos_38" in raw
        else {"#CHROM": "chrom", "POS": "pos", "REF": "ref", "ALT": "alt"}
    )
    frame = raw.rename(columns=names).copy()
    for column in KEYS + ["variant38"]:
        frame[column] = frame[column].str.strip()
    frame["chrom"] = frame["chrom"].str.replace(r"^chr", "", regex=True).replace({"M": "MT"})
    frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype("int64")
    frame[["ref", "alt"]] = frame[["ref", "alt"]].apply(lambda column: column.str.upper())
    valid = (
        frame["pos"].ge(65)
        & frame["ref"].isin(list("ACGT"))
        & frame["alt"].isin(list("ACGT"))
        & frame["ref"].ne(frame["alt"])
    )
    expected = (
        frame["chrom"] + "_" + frame["pos"].astype(str) + "_" + frame["ref"] + "/" + frame["alt"]
    )
    valid &= frame["variant38"].str.replace(r"^chr", "", regex=True).eq(expected)
    if not valid.all():
        raise ValueError(
            f"{path}: invalid SNV or coordinate at lines {(frame.index[~valid] + 2).tolist()[:10]}"
        )
    variants = frame[KEYS].drop_duplicates().reset_index(drop=True)
    return variants, {
        "input_rows": len(raw),
        "unique_variants": len(variants),
        "duplicate_variant_rows": len(raw) - len(variants),
        "unique_sites": len(variants.drop_duplicates(["chrom", "pos"])),
    }


def index_sites(variants):
    """
    Desc:
        Index sorted genomic positions by chromosome.
    Args:
        variants, Value supplied for variants.
    """
    frame = pd.DataFrame(variants, columns=KEYS)
    sites = frame[["chrom", "pos"]].drop_duplicates().sort_values(["chrom", "pos"])
    return {chrom: group["pos"].to_numpy() for chrom, group in sites.groupby("chrom")}


def neighbor_count(positions, pos, flank):
    """
    Desc:
        Count target sites within each query interval.
    Args:
        positions, Value supplied for positions.
        pos, Value supplied for pos.
        flank, Context flank length in bases.
    """
    positions = np.asarray(positions)
    return np.searchsorted(positions, np.asarray(pos) + 2 * flank, side="right") - np.searchsorted(
        positions, np.asarray(pos) - 2 * flank, side="left"
    )


def merged_intervals(variants, flank):
    """
    Desc:
        Merge overlapping genomic intervals by chromosome.
    Args:
        variants, Value supplied for variants.
        flank, Context flank length in bases.
    """
    result = {}
    for chrom, positions in index_sites(variants).items():
        merged = []
        for pos in positions:
            start, end = pos - flank, pos + flank
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result[chrom] = merged
    return result


def covered_bp(intervals):
    """
    Desc:
        Calculate the total number of bases covered by merged intervals.
    Args:
        intervals, Value supplied for intervals.
    """
    return sum(end - start + 1 for items in intervals.values() for start, end in items)


def intersection_bp(left, right):
    """
    Desc:
        Calculate the shared coverage of two interval collections.
    Args:
        left, Value supplied for left.
        right, Value supplied for right.
    """
    total = 0
    for chrom in left.keys() & right.keys():
        a, b = left[chrom], right[chrom]
        i = j = 0
        while i < len(a) and j < len(b):
            start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
            total += max(0, end - start + 1)
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
    return total


def directional_stats(source, target, group, source_name, target_name, flank):
    """
    Desc:
        Calculate directional overlap statistics and per-variant details.
    Args:
        source, Source variant collection.
        target, Target variant collection.
        group, Variant-label group.
        source_name, Value supplied for source_name.
        target_name, Value supplied for target_name.
        flank, Context flank length in bases.
    """
    source = pd.DataFrame(source, columns=KEYS).drop_duplicates()
    target = pd.DataFrame(target, columns=KEYS).drop_duplicates()
    target_index = index_sites(target)
    details = source.sort_values(KEYS).reset_index(drop=True).copy()
    details["overlapping_target_unique_sites"] = 0

    for chrom, rows in details.groupby("chrom"):
        details.loc[rows.index, "overlapping_target_unique_sites"] = neighbor_count(
            target_index.get(chrom, np.array([], dtype="int64")), rows["pos"].to_numpy(), flank
        )
    details["overlaps"] = details["overlapping_target_unique_sites"].gt(0).astype(int)
    details["variant38"] = (
        details["chrom"]
        + "_"
        + details["pos"].astype(str)
        + "_"
        + details["ref"]
        + "/"
        + details["alt"]
    )
    details["context_start"] = details["pos"] - flank
    details["context_end"] = details["pos"] + flank
    details = details.assign(group=group, source=source_name, target=target_name)
    sites = details.drop_duplicates(["chrom", "pos"])
    hits, site_hits = int(details["overlaps"].sum()), int(sites["overlaps"].sum())
    summary = {
        "group": group,
        "source": source_name,
        "target": target_name,
        "source_unique_variants": len(source),
        "target_unique_variants": len(target),
        "overlapping_source_variants": hits,
        "overlap_percent": 100 * hits / len(source) if len(source) else 0,
        "source_unique_sites": len(sites),
        "target_unique_sites": sum(len(v) for v in target_index.values()),
        "overlapping_source_sites": site_hits,
        "site_overlap_percent": 100 * site_hits / len(sites) if len(sites) else 0,
        "overlapping_unique_site_pairs": int(sites["overlapping_target_unique_sites"].sum()),
    }
    columns = [
        "group",
        "source",
        "target",
        "variant38",
        "chrom",
        "pos",
        "context_start",
        "context_end",
        "overlaps",
        "overlapping_target_unique_sites",
    ]
    return summary, details[columns]


def main():
    """
    Desc:
        Run the command-line workflow.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "statistics")
    for dataset in ("missense", "synonymous", "noncoding"):
        for group in ("benign", "pathogenic"):
            parser.add_argument(f"--{dataset}-{group}", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        dataset: {group: getattr(args, f"{dataset}_{group}") for group in ("benign", "pathogenic")}
        for dataset in ("missense", "synonymous", "noncoding")
    }
    output = args.output_dir.resolve()
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
    flank = 64
    data, qc, provenance = {}, [], []
    for name, paths in inputs.items():
        for group, path in paths.items():
            variants, quality = load_variants(path)
            data[(name, group)] = variants
            qc.append({"dataset": name, "group": group, **quality})
            provenance.append(
                {"dataset": name, "group": group, "path": str(path), "sha256": sha256(path)}
            )
            logging.info("%s / %s: %s", name, group, quality)

        data[(name, "combined")] = (
            pd.concat([data[(name, "benign")], data[(name, "pathogenic")]], ignore_index=True)
            .drop_duplicates(KEYS)
            .reset_index(drop=True)
        )
        combined = data[(name, "combined")]
        qc.append(
            {
                "dataset": name,
                "group": "combined",
                "input_rows": sum(r["input_rows"] for r in qc if r["dataset"] == name),
                "unique_variants": len(combined),
                "duplicate_variant_rows": sum(r["input_rows"] for r in qc if r["dataset"] == name)
                - len(combined),
                "unique_sites": len(combined.drop_duplicates(["chrom", "pos"])),
            }
        )
    summary, coverage, detail_tables = [], [], []
    for group in GROUPS:
        for name in ("missense", "noncoding"):
            target, syn = data[(name, group)], data[("synonymous", group)]
            for source_name, target_name, source_set, target_set in (
                ("synonymous", name, syn, target),
                (name, "synonymous", target, syn),
            ):
                result, detail = directional_stats(
                    source_set, target_set, group, source_name, target_name, flank
                )
                summary.append(result)
                detail_tables.append(detail)
                logging.info(
                    "%s / %s → %s: %d/%d = %.6f%%",
                    group,
                    source_name,
                    target_name,
                    result["overlapping_source_variants"],
                    len(source_set),
                    result["overlap_percent"],
                )
            assert (
                summary[-1]["overlapping_unique_site_pairs"]
                == summary[-2]["overlapping_unique_site_pairs"]
            )
            a, b = merged_intervals(target, flank), merged_intervals(syn, flank)
            a_bp, b_bp, shared = covered_bp(a), covered_bp(b), intersection_bp(a, b)
            union = a_bp + b_bp - shared
            coverage.append(
                {
                    "group": group,
                    "dataset_a": name,
                    "dataset_b": "synonymous",
                    "a_covered_bp": int(a_bp),
                    "b_covered_bp": int(b_bp),
                    "intersection_bp": int(shared),
                    "union_bp": int(union),
                    "jaccard_percent": 100 * shared / union if union else 0,
                    "a_covered_by_b_percent": 100 * shared / a_bp if a_bp else 0,
                    "b_covered_by_a_percent": 100 * shared / b_bp if b_bp else 0,
                }
            )
    summary_table = pd.DataFrame(summary)
    details = pd.concat(detail_tables, ignore_index=True)

    audited = details.groupby(["group", "source", "target"])["overlaps"].agg(["size", "sum"])
    expected = summary_table.set_index(["group", "source", "target"]).sort_index()
    assert audited["size"].eq(expected["source_unique_variants"]).all()
    assert audited["sum"].eq(expected["overlapping_source_variants"]).all()
    logging.info("PASS: all 12 summaries agree with the per-variant details")
    summary_table.to_csv(output / "overlap_summary.tsv", sep="\t", index=False)
    summary_table.loc[summary_table["target"].eq("synonymous")].to_csv(
        output / "overlap_with_synonymous.tsv", sep="\t", index=False
    )
    details.to_csv(output / "per_variant_overlap.tsv", sep="\t", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage_summary.tsv", sep="\t", index=False)
    pd.DataFrame(qc).to_csv(output / "dataset_qc.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "genome_build": "hg38 (input annotation)",
        "coordinate_system": "1-based inclusive",
        "flank_bp": flank,
        "context_length_bp": 129,
        "overlap_rule": "Same chromosome; abs(POS_A - POS_B) <= 128; at least 1 shared bp",
        "primary_unit": "Unique (chromosome, position, REF, ALT) variant; overlap_percent denominator is source",
        "secondary_units": "Unique genomic sites; merged genomic base-pair coverage",
        "combined_rule": "Union benign and pathogenic variant keys, then recompute overlap",
        "strand_rule": "Ignore strand; measure genomic overlap, not DNA sequence identity",
        "sequence_columns": "Not read or validated; contexts derived only from genomic positions",
        "boundaries": "All positions >=65 verified; chromosome-end clipping not applied (no assembly lengths supplied)",
        "inputs": provenance,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    logging.info("Analysis complete：%s", output)


if __name__ == "__main__":
    main()
