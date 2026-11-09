"""
@description: build reference and alternate codon contexts for synonymous SNVs
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings("ignore")


def get_context_for_codon(cds_pos: int, codon_context: int, cds_seq: str, alt: str, strand: int):
    """
    Desc:
        Build reference and alternate codon-context sequences.
    Args:
        cds_pos, Value supplied for cds_pos.
        codon_context, Value supplied for codon_context.
        cds_seq, Value supplied for cds_seq.
        alt, Alternate allele.
        strand, Value supplied for strand.
    """
    codon_pos = (cds_pos - 1) // 3 + 1  # 1-based codon index
    mut_pos_in_codon = (cds_pos - 1) % 3
    ref_seq = get_codon_context(codon_pos, codon_context, cds_seq)
    alt_seq = generate_alter_codon_seq(ref_seq, alt, strand, codon_context, mut_pos_in_codon)
    return ref_seq, alt_seq


def get_codon_context(codon_pos: int, context: int, seq: str):
    """
    Desc:
        Extract a fixed number of codons around the selected codon.
    Args:
        codon_pos, Value supplied for codon_pos.
        context, Value supplied for context.
        seq, Value supplied for seq.
    """
    codon_count = len(seq) // 3
    codons = [seq[i * 3 : (i + 1) * 3] for i in range(codon_count)]

    start = codon_pos - context - 1  # 0-based
    end = codon_pos + context

    ref_seq = []

    if start < 0:
        missing_left = -start
        for i in range(min(missing_left, codon_count)):
            ref_seq.append(codons[i])
        start = 0

    if start < end and start < codon_count:
        for i in range(start, min(end, codon_count)):
            ref_seq.append(codons[i])

    if end > codon_count:
        missing_right = end - codon_count
        if missing_right > 0:
            take = min(missing_right, codon_count)
            for i in range(codon_count - take, codon_count):
                ref_seq.append(codons[i])

    return "".join(ref_seq)


def generate_alter_codon_seq(
    ref_seq: str, alt: str, strand: int, context: int, mut_pos_in_codon: int
) -> str:
    """
    Desc:
        Replace the selected codon base with the alternate allele.
    Args:
        ref_seq, Value supplied for ref_seq.
        alt, Alternate allele.
        strand, Value supplied for strand.
        context, Value supplied for context.
        mut_pos_in_codon, Value supplied for mut_pos_in_codon.
    """
    alt = alt.upper()
    if strand < 0:
        alt = complement(alt)

    cs = list(ref_seq)
    center_start = context * 3
    replace_index = center_start + mut_pos_in_codon

    if replace_index < 0 or replace_index >= len(cs):
        raise ValueError("mutation position outside context range")

    cs[replace_index] = alt
    return "".join(cs)


def complement(base: str) -> str:
    """
    Desc:
        Return the complementary nucleotide sequence.
    Args:
        base, Value supplied for base.
    """
    comp_map = str.maketrans("ATCG", "TAGC")
    return base.translate(comp_map)


def process_variant_df(
    df_variants, df_transcript, codon_context=21, transcript_column_name="transcript_id"
):
    """
    Desc:
        Generate codon contexts for all valid transcript-matched variants.
    Args:
        df_variants, Variant data frame.
        df_transcript, Transcript metadata data frame.
        codon_context, Value supplied for codon_context.
        transcript_column_name, Value supplied for transcript_column_name.
    """

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    df_variant = df_variants.groupby(transcript_column_name)
    results = []

    for tid, tdf in df_variant:
        transcript_match = df_transcript[df_transcript[transcript_column_name] == tid]
        if transcript_match.empty:
            logging.warning(
                f"No transcript metadata found for transcript_id {tid}; skipping {len(tdf)} rows"
            )
            continue
        transcript_meta = transcript_match.iloc[0]
        cds_seq = transcript_meta["coding_sequence"]
        strand = transcript_meta["strand"]
        if pd.isna(strand) or pd.isna(cds_seq) or cds_seq == "":
            logging.warning(f"Transcript {tid} has no strand or CDS; skipping {len(tdf)} rows")
            continue
        strand = int(strand)
        for idx, row in tdf.iterrows():
            row_data = row.copy()
            alt = row["ALT"]
            cds_pos = row["cds_pos"]
            if pd.isna(alt) or pd.isna(cds_pos):
                logging.warning(f"Transcript {tid} row {idx} has no ALT or CDS position; skipping")
                continue
            cds_pos = int(cds_pos)

            try:
                ref_seq, alt_seq = get_context_for_codon(
                    cds_pos, codon_context, cds_seq, alt, strand
                )
                row_data["codon_ref"] = ref_seq
                row_data["codon_alt"] = alt_seq
                results.append(row_data)
            except Exception as e:
                logging.warning(f"Transcript {tid} position {cds_pos} failed: {e}; skipping")
                continue

    df_result = pd.DataFrame(results) if results else pd.DataFrame()
    if not df_result.empty:
        df_result = df_result.reset_index(drop=True)
        print(f"Successfully processed {len(df_result)}/{len(df_variants)} rows")

        skipped = len(df_variants) - len(df_result)
        if skipped > 0:
            print(f"Skipped {skipped} rows because of missing values or errors")
    else:
        print("No records were processed successfully")
    return df_result


def main(args):
    # 1 load transcript info
    """
    Desc:
        Run the command-line workflow.
    Args:
        args, Parsed command-line arguments.
    """
    transcript_path = args.transcript_info
    df_transcript = pd.read_csv(
        transcript_path,
        sep="\t",
        usecols=["transcript_id", "strand", "refseq_id", "coding_sequence"],
    )

    task_dir = args.task_dir

    usecols = ["#CHROM", "POS", "REF", "ALT", "Variant38", "transcript_id", "cds_pos"]

    df = pd.read_csv(f"{task_dir}/input.cds.txt", sep="\t", usecols=usecols)
    df_codon = process_variant_df(
        df_variants=df,
        df_transcript=df_transcript,
        codon_context=21,
        transcript_column_name="transcript_id",
    )
    df_codon.to_csv(f"{task_dir}/input.codon.txt", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from biological data.")
    parser.add_argument(
        "--task_dir", type=str, required=True, help="Path to save the output files."
    )
    parser.add_argument(
        "--transcript_info",
        required=True,
        help="TSV containing transcript_id, strand, refseq_id, and coding_sequence.",
    )
    args = parser.parse_args()
    main(args)
