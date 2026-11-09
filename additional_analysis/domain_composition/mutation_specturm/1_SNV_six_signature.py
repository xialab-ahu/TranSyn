"""
@description: calculate six-class SNV spectra across CDS positions
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from signature_utils import *
import re
from plot_six_signature import BINS, plot_statistics, plot_from_file

TRANSCRIPTS = (
    PROJECT / "B-feature/00-total-feature-pool/rna_protein_sequence/transcript_id_info_summary.txt"
)
MISSENSE = PROJECT / "A-prepare-dataset/b-output/dataset_missense"
SYNONYMOUS = ROOT / "9_synonymous_dataset/F-final-result/Version1"
INPUTS = {
    "missense": {
        "benign": MISSENSE
        / "missense.bn.with_transcript.51936.sanity.check.3.with.codon.info.noDMq.txt",
        "pathogenic": MISSENSE
        / "missense.pt.with_transcript.51936.sanity.check.3.with.codon.info.noDMq.txt",
    },
    "synonymous": {
        "benign": SYNONYMOUS / "train_neg_1181_with_codon_info_flank_with_rept.txt",
        "pathogenic": SYNONYMOUS / "train_pos_1181_with_codon_info_flank_with_rept.txt",
    },
}


def transcript_index(logger):
    """
    Desc:
        Execute the transcript index operation.
    Args:
        logger, Value supplied for logger.
    """
    index = {}
    for line, row in rows(TRANSCRIPTS):
        try:
            seq = row["coding_sequence"].upper()
            length = int(row["cds_length"])
            strand = int(row["strand"])
            if length <= 0 or len(seq) != length or strand not in (-1, 1):
                raise ValueError("invalid_transcript_length_or_strand")
            value = (seq, length, strand)
            for key in [row["transcript_id"]] + re.split(r"[;,|\s]+", row["refseq_id"]):
                if key:
                    index.setdefault(key, set()).add(value)
        except (ValueError, KeyError, AttributeError) as exc:
            logger.warning("%s:%s\t%s", TRANSCRIPTS, line, exc)
    return index


def collect(path, dataset, index, logger):
    """
    Desc:
        Execute the collect operation.
    Args:
        path, Input filesystem path.
        dataset, Value supplied for dataset.
        index, Value supplied for index.
        logger, Value supplied for logger.
    """
    accepted, total = [], 0
    for line, row in rows(path):
        total += 1
        try:
            ref = row["REF" if dataset == "missense" else "ref"].upper()
            alt = row["ALT" if dataset == "missense" else "alt"].upper()
            signature, _ = canonical(ref, alt)
            key = row["refseq_id" if dataset == "missense" else "transcript_id"]
            candidates = index.get(key, set())
            if not candidates:
                raise ValueError("transcript_not_found")
            if len(candidates) != 1:
                raise ValueError("ambiguous_transcript_annotation")
            seq, length, strand = next(iter(candidates))
            pos = int(row["cds_pos"])
            if not 1 <= pos <= length:
                raise ValueError("CDS_position_out_of_range")
            transcript_ref = ref if strand == 1 else ref.translate(COMP)
            if seq[pos - 1] != transcript_ref:
                raise ValueError("CDS_reference_mismatch")

            bin_id = (5 * pos - 1) // length
            accepted.append((signature, bin_id))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("%s:%s\t%s\t%s", path, line, exc, row.get("cds_pos", ""))
    return accepted, {
        "dataset": dataset,
        "group": "",
        "input_file": str(path),
        "total": total,
        "accepted": len(accepted),
        "excluded": total - len(accepted),
    }


def main():
    """
    Desc:
        Run the command-line workflow.
    """
    destination, logger = setup("six_signature")
    index = transcript_index(logger)
    tables, summaries = [], []
    for dataset, files in INPUTS.items():
        records = {}
        for group, path in files.items():
            records[group], summary = collect(path, dataset, index, logger)
            summary["group"] = group
            summaries.append(summary)
        records["combined"] = records["benign"] + records["pathogenic"]
        for group in GROUPS:
            counts = np.zeros((6, 5), dtype=int)
            for signature, bin_id in records[group]:
                counts[SIGNATURES.index(signature), bin_id] += 1
            totals = counts.sum(axis=1)
            fractions = np.divide(
                counts, totals[:, None], out=np.full((6, 5), np.nan), where=totals[:, None] > 0
            )
            for i, signature in enumerate(SIGNATURES):
                for j, label in enumerate(BINS):
                    tables.append(
                        {
                            "dataset": dataset,
                            "group": group,
                            "signature": signature,
                            "cds_bin": label,
                            "count": counts[i, j],
                            "signature_total": totals[i],
                            "fraction": fractions[i, j],
                            "expected_fraction": 0.2,
                            "deviation_percentage_points": (fractions[i, j] - 0.2) * 100,
                        }
                    )
    pd.DataFrame(tables).to_csv(destination / "cds_six_statistics.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(destination / "record_summary.tsv", sep="\t", index=False)
    plot_statistics(pd.DataFrame(tables), destination)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze six SNV classes across CDS positions")
    parser.add_argument(
        "--plot-only", action="store_true", help="Skip counting and plot an existing table"
    )
    parser.add_argument(
        "--statistics", type=Path, default=BASE / "results/six_signature/cds_six_statistics.tsv"
    )
    parser.add_argument("--output-dir", type=Path, help="Figure directory in plot-only mode")
    args = parser.parse_args()
    if args.plot_only:
        plot_from_file(args.statistics, args.output_dir or args.statistics.parent)
    else:
        if args.output_dir is not None:
            parser.error("--output-dir requires --plot-only")
        main()
