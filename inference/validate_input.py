#!/usr/bin/env python3
"""Validate and normalize a GRCh38 synonymous-SNV input table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Validate columns, alleles, coordinates, duplicates, and optional FASTA bases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path,
                        help="Indexed GRCh38 FASTA used for reference-allele checks")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, sep="\t", dtype=str)
    frame = frame.rename(columns={"CHROM": "#CHROM", "chrom": "#CHROM",
                                  "pos": "POS", "ref": "REF", "alt": "ALT"})
    required = ["#CHROM", "POS", "REF", "ALT"]
    if missing := set(required).difference(frame.columns):
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    frame = frame[required].copy()
    frame["#CHROM"] = frame["#CHROM"].str.removeprefix("chr")
    frame["POS"] = pd.to_numeric(frame["POS"], errors="raise").astype(int)
    frame[["REF", "ALT"]] = frame[["REF", "ALT"]].apply(lambda column: column.str.upper())
    chromosomes = {str(value) for value in range(1, 23)} | {"X", "Y"}
    valid = (frame["#CHROM"].isin(chromosomes) & frame["POS"].gt(0)
             & frame["REF"].isin(list("ACGT")) & frame["ALT"].isin(list("ACGT"))
             & frame["REF"].ne(frame["ALT"]))
    if not valid.all():
        raise ValueError(f"Unsupported SNV rows (one-based file lines): {(frame.index[~valid] + 2).tolist()}")
    if frame.duplicated(required).any():
        raise ValueError("Duplicate variants are not supported")
    if args.reference_fasta:
        try:
            from pyfaidx import Fasta
        except ImportError as error:
            raise RuntimeError("Install pyfaidx to perform reference checks") from error
        reference = Fasta(str(args.reference_fasta), as_raw=True, sequence_always_upper=True)
        mismatches = []
        for index, row in frame.iterrows():
            candidates = (row["#CHROM"], f"chr{row['#CHROM']}")
            contig = next((name for name in candidates if name in reference), None)
            observed = reference[contig][row["POS"] - 1:row["POS"]] if contig else ""
            if observed != row["REF"]:
                mismatches.append(index + 2)
        if mismatches:
            raise ValueError(f"GRCh38 reference-allele mismatches at file lines: {mismatches}")
    frame["Variant38"] = (frame["#CHROM"] + "_" + frame["POS"].astype(str)
                          + "_" + frame["REF"] + "/" + frame["ALT"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t", index=False)
    print(f"PASS: {len(frame)} unique GRCh38 biallelic SNVs")


if __name__ == "__main__":
    main()
