"""
@description: plot context-overlap statistics
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
COLORS = {"missense": "#d86261", "noncoding": "#5ebddc", "synonymous": "#e0e0e0"}
LABELS = {"missense": "Missense", "noncoding": "Noncoding", "synonymous": "Synonymous"}
GROUPS = ("benign", "pathogenic", "combined")


def plot_overlap(summary_path, output_dir, unit):
    """
    Desc:
        Execute the plot overlap operation.
    Args:
        summary_path, Value supplied for summary_path.
        output_dir, Output directory.
        unit, Value supplied for unit.
    """
    table = pd.read_csv(summary_path, sep="\t").set_index(["group", "source", "target"])
    reverse = table.index.get_level_values("source").unique().tolist() == ["synonymous"]

    is_extent = (
        "metric" in table.columns and table["metric"].eq("mean_max_overlap_fraction_all").all()
    )

    if unit == "variants":
        percent_key, numerator_key, denominator_key = (
            "overlap_percent",
            "overlapping_source_variants",
            "source_unique_variants",
        )
    else:
        percent_key, numerator_key, denominator_key = (
            "site_overlap_percent",
            "overlapping_source_sites",
            "source_unique_sites",
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    names = ("noncoding", "missense")

    def result_for(group, name):
        """
        Desc:
            Execute the result for operation.
        Args:
            group, Variant-label group.
            name, Output or task name.
        """
        key = (group, "synonymous", name) if reverse else (group, name, "synonymous")
        return table.loc[key]

    selected = [result_for(group, name) for group in GROUPS for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3), sharey=True)
    ymax = max(1, max(float(row[percent_key]) for row in selected) * 1.35)
    for ax, group in zip(axes, GROUPS):
        results = [result_for(group, name) for name in names]
        values = [float(result[percent_key]) for result in results]
        ax.bar(
            [0, 1],
            values,
            width=0.55,
            color=[COLORS[name] for name in names],
            edgecolor="#666666",
            linewidth=0.6,
            zorder=3,
        )
        for x, value, result in zip((0, 1), values, results):
            numerator, denominator = int(result[numerator_key]), int(result[denominator_key])
            annotation = (
                f"{value:.2f}%" if is_extent else f"{value:.2f}%\n({numerator:,}/{denominator:,})"
            )
            ax.text(x, value + ymax * 0.025, annotation, ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], [LABELS[name] for name in names])
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.65, 1.65)
        ax.grid(axis="y", color="#ececec", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(group.capitalize(), fontsize=12, pad=14)
    ylabel = (
        "Mean normalized context overlap\nwith synonymous contexts (%)"
        if is_extent
        else f"{unit.capitalize()} with context overlap\nwith synonymous contexts (%)"
    )
    if reverse:
        ylabel = "Mean overlap of synonymous contexts (%)"
    axes[0].set_ylabel(ylabel, labelpad=10)
    fig.tight_layout(rect=(0.015, 0.13, 0.995, 0.91))
    for suffix in ["png"]:
        destination = output_dir / f"context_overlap_{unit}.{suffix}"
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        logging.info("Saved %s", destination)
    plt.close(fig)


def main():
    """
    Desc:
        Run the command-line workflow.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=BASE / "statistics/overlap_with_synonymous.tsv"
    )
    parser.add_argument("--output-dir", type=Path, default=BASE / "figures")
    parser.add_argument("--unit", choices=("variants", "sites", "both"), default="both")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "plotting.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    for unit in ("variants", "sites") if args.unit == "both" else (args.unit,):
        plot_overlap(args.summary, args.output_dir, unit)


if __name__ == "__main__":
    main()
