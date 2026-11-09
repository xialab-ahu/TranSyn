"""
@description: calculate trinucleotide 96-channel mutation spectra
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from signature_utils import *
from itertools import product
from plot_tri_96 import plot_statistics, plot_from_file

FLANKS = list(product("ATCG", repeat=2))

NONCODING = PROJECT / "/noncoding_dataset/"
SYNONYMOUS = ROOT / "/synonymous_dataset/"
INPUTS = {
    "noncoding": {
        "benign": NONCODING / "noncoding.bn.txt",
        "pathogenic": NONCODING / "noncoding.pt.txt",
    },
    "synonymous": {
        "benign": SYNONYMOUS / "train_neg.txt",
        "pathogenic": SYNONYMOUS / "train_pos.txt",
    },
}
FASTA_PATH = ROOT / "/reference_genome/hg38.fa"


def open_genome(path):
    """
    Desc:
        Execute the open genome operation.
    Args:
        path, Input filesystem path.
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        import sys

        cached = ROOT / "miniconda3/pkgs/pyfaidx-0.5.3-py36_0/lib/python3.6/site-packages"
        if not (cached / "pyfaidx").is_dir():
            raise ImportError("Install pyfaidx in the analysis environment")
        sys.path.append(str(cached))
        from pyfaidx import Fasta
    if not Path(str(path) + ".fai").is_file():
        raise FileNotFoundError(f"An existing FASTA index is required：{path}.fai")
    return Fasta(
        str(path), as_raw=True, sequence_always_upper=True, rebuild=False, build_index=False
    )


def collect(path, fasta, sequence_output, logger):
    """
    Desc:
        Execute the collect operation.
    Args:
        path, Input filesystem path.
        fasta, Value supplied for fasta.
        sequence_output, Value supplied for sequence_output.
        logger, Value supplied for logger.
    """
    accepted, total = [], 0
    keys = set(fasta.keys())
    with sequence_output.open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "source_line",
                "ID",
                "#CHROM",
                "POS",
                "REF",
                "ALT",
                "Ref_Trinucleotide",
                "signature",
                "context",
                "channel",
            ]
        )
        for line, row in rows(path):
            total += 1
            variant_id = "unknown"
            try:
                chrom = row["#CHROM"].strip()
                ref, alt = row["REF"].strip().upper(), row["ALT"].strip().upper()
                pos = int(row["POS"])
                variant_id = f"{chrom}_{pos}_{ref}/{alt}"
                signature, reverse = canonical(ref, alt)
                if not chrom or pos < 1:
                    raise ValueError("invalid_chromosome_or_position")
                if chrom not in keys:
                    chrom = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
                if chrom not in keys:
                    raise ValueError("chromosome_not_found")

                start, end = pos - 2, pos + 1
                if start < 0 or end > len(fasta[chrom]):
                    raise ValueError("trinucleotide_out_of_bounds")
                tri = fasta[chrom][start:end]
                if len(tri) != 3 or any(base not in "ACGT" for base in tri):
                    raise ValueError("invalid_trinucleotide")
                if tri[1] != ref:
                    raise ValueError(f"reference_mismatch: expected={ref}, observed={tri[1]}")
                original_tri = tri
                if reverse:
                    tri = tri.translate(COMP)[::-1]

                channel = f"{tri[0]}[{signature}]{tri[2]}"
                writer.writerow(
                    [
                        line,
                        variant_id,
                        row["#CHROM"],
                        pos,
                        ref,
                        alt,
                        original_tri,
                        signature,
                        tri,
                        channel,
                    ]
                )
                accepted.append((signature, tri))
            except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                logger.warning("%s:%s\tID=%s\t%s", path, line, variant_id, exc)
    return accepted, {
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
    destination, logger = setup("tri_96_signature")
    records, summaries, tables = {}, [], []
    import json
    import sys

    with open_genome(FASTA_PATH) as fasta:
        for dataset, files in INPUTS.items():
            records[dataset] = {}
            for group, path in files.items():
                sequence_output = destination / f"{dataset}_{group}_contexts.tsv"
                records[dataset][group], summary = collect(path, fasta, sequence_output, logger)
                summaries.append({"dataset": dataset, "group": group, **summary})
            records[dataset]["combined"] = (
                records[dataset]["benign"] + records[dataset]["pathogenic"]
            )
    metadata = {
        "pyfaidx_version": sys.modules["pyfaidx"].__version__,
        "fasta": str(FASTA_PATH),
        "context_length": 3,
        "flank_length": 1,
        "variant_index": 1,
        "coordinate_system": "1-based POS; 0-based half-open extraction",
        "inputs": {d: {g: str(f) for g, f in fs.items()} for d, fs in INPUTS.items()},
    }
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    for group in GROUPS:
        for dataset in INPUTS:
            counter = pd.Series(records[dataset][group], dtype=object).value_counts().to_dict()
            for signature in SIGNATURES:
                for left, right in FLANKS:
                    tri = left + signature[0] + right
                    tables.append(
                        {
                            "dataset": dataset,
                            "group": group,
                            "signature": signature,
                            "context": tri,
                            "channel": f"{left}[{signature}]{right}",
                            "count": counter.get((signature, tri), 0),
                        }
                    )
    pd.DataFrame(tables).to_csv(destination / "tri_96_statistics.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(destination / "record_summary.tsv", sep="\t", index=False)
    plot_statistics(pd.DataFrame(tables), destination)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze trinucleotide 96-channel mutation spectra"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Skip extraction and plot an existing table"
    )
    parser.add_argument(
        "--statistics", type=Path, default=BASE / "results/tri_96_signature/tri_96_statistics.tsv"
    )
    parser.add_argument("--output-dir", type=Path, help="Figure directory in plot-only mode")
    args = parser.parse_args()
    if args.plot_only:
        plot_from_file(args.statistics, args.output_dir or args.statistics.parent)
    else:
        if args.output_dir is not None:
            parser.error("--output-dir requires --plot-only")
        main()
