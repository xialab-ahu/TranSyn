#!/usr/bin/env python3
"""Prepare noncoding and missense source-domain SNVs from licensed inputs.

Splice-variant preprocessing is intentionally excluded. HGMD is read from a
user-supplied local file and is never redistributed with TranSyn.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

BASES = frozenset("ACGT")
CHROMOSOMES = frozenset([str(value) for value in range(1, 23)] + ["X", "Y"])
PATHOGENIC = frozenset({"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"})
BENIGN = frozenset({"Benign", "Likely_benign", "Benign/Likely_benign"})
CONSEQUENCES = {
    "noncoding": {"SO:0001627|intron_variant", "SO:0001631|upstream_gene_variant",
                  "SO:0001632|downstream_gene_variant", "SO:0001623|5_prime_UTR_variant",
                  "SO:0001624|3_prime_UTR_variant", "SO:0001619|non_coding_transcript_variant"},
    "missense": {"SO:0001583|missense_variant"},
}


def variant_id(frame: pd.DataFrame) -> pd.Series:
    """Return canonical GRCh38 identifiers in CHROM_POS_REF/ALT form."""
    return (frame["chrom"].astype(str) + "_" + frame["pos"].astype(str) + "_"
            + frame["ref"] + "/" + frame["alt"])


def validate_snvs(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize GRCh38 SNVs and remove unsupported or malformed records."""
    result = frame.copy()
    result["chrom"] = result["chrom"].astype(str).str.removeprefix("chr")
    result["pos"] = pd.to_numeric(result["pos"], errors="coerce").astype("Int64")
    result["ref"] = result["ref"].astype(str).str.upper()
    result["alt"] = result["alt"].astype(str).str.upper()
    valid = (result["chrom"].isin(CHROMOSOMES) & result["pos"].notna()
             & result["ref"].isin(BASES) & result["alt"].isin(BASES)
             & result["ref"].ne(result["alt"]))
    logging.info("%s: retained %d/%d valid SNVs", source, valid.sum(), len(result))
    result = result.loc[valid].copy()
    result["pos"] = result["pos"].astype(int)
    result["variant_hg38"] = variant_id(result)
    return result.drop_duplicates("variant_hg38")


def consequence_mask(values: pd.Series, domain: str) -> pd.Series:
    """Match any comma-separated ClinVar molecular consequence."""
    accepted = CONSEQUENCES[domain]
    return values.fillna("").apply(
        lambda value: bool(accepted.intersection(part.strip() for part in value.split(","))))


def load_clinvar(path: Path, domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a parsed GRCh38 ClinVar table and split supported labels."""
    frame = pd.read_csv(path, sep="\t", dtype=str).rename(columns={
        "#CHROM": "chrom", "POS": "pos", "REF": "ref", "ALT": "alt",
        "MC": "consequence", "CLNSIG": "significance"})
    required = {"chrom", "pos", "ref", "alt", "consequence", "significance"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"ClinVar input is missing columns: {sorted(missing)}")
    frame = validate_snvs(frame, "ClinVar")
    frame = frame.loc[consequence_mask(frame["consequence"], domain)].copy()
    pathogenic = frame.loc[frame["significance"].isin(PATHOGENIC)].copy()
    benign = frame.loc[frame["significance"].isin(BENIGN)].copy()
    pathogenic["source"], benign["source"] = "ClinVar", "ClinVar"
    return pathogenic, benign


def load_hgmd(path: Path, domain: str, include_dmq: bool) -> pd.DataFrame:
    """Load licensed HGMD records from a path supplied by the user."""
    frame = pd.read_csv(path, dtype=str, low_memory=False).rename(columns={
        "chrom_hg38": "chrom", "pos_hg38": "pos", "ref_hg38": "ref",
        "alt_hg38": "alt", "mutype": "consequence"})
    required = {"chrom", "pos", "ref", "alt", "consequence", "tag"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"HGMD input is missing columns: {sorted(missing)}")
    tags = {"DM", "DM?"} if include_dmq else {"DM"}
    expected = "missense" if domain == "missense" else "regulatory"
    frame = frame.loc[frame["tag"].isin(tags) & frame["consequence"].eq(expected)]
    frame = validate_snvs(frame, "HGMD")
    frame["source"] = "HGMD"
    return frame


def main() -> None:
    """Prepare one source domain and write record-level data plus QC counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinvar", type=Path, required=True, help="Parsed GRCh38 ClinVar TSV")
    parser.add_argument("--hgmd", type=Path, required=True, help="Licensed HGMD CSV")
    parser.add_argument("--domain", choices=sorted(CONSEQUENCES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-variants", type=Path, help="TSV containing variant_hg38")
    parser.add_argument("--include-dmq", action="store_true", help="Also include HGMD DM? records")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clinvar_pathogenic, benign = load_clinvar(args.clinvar, args.domain)
    pathogenic = pd.concat(
        [clinvar_pathogenic, load_hgmd(args.hgmd, args.domain, args.include_dmq)],
        ignore_index=True).drop_duplicates("variant_hg38")
    conflicts = set(pathogenic["variant_hg38"]) & set(benign["variant_hg38"])
    pathogenic = pathogenic.loc[~pathogenic["variant_hg38"].isin(conflicts)].copy()
    benign = benign.loc[~benign["variant_hg38"].isin(conflicts)].copy()
    if args.exclude_variants:
        excluded = set(pd.read_csv(args.exclude_variants, sep="\t")["variant_hg38"])
        pathogenic = pathogenic.loc[~pathogenic["variant_hg38"].isin(excluded)]
        benign = benign.loc[~benign["variant_hg38"].isin(excluded)]
    columns = ["chrom", "pos", "ref", "alt", "variant_hg38", "source"]
    pathogenic[columns].to_csv(args.output_dir / "pathogenic.tsv", sep="\t", index=False)
    benign[columns].to_csv(args.output_dir / "benign.tsv", sep="\t", index=False)
    pd.DataFrame([{"class": "pathogenic", "count": len(pathogenic)},
                  {"class": "benign", "count": len(benign)}]).to_csv(
                      args.output_dir / "qc_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
