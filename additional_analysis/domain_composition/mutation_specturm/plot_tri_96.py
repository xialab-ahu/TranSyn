"""
@description: plot trinucleotide 96-channel mutation spectra
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from itertools import product
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from signature_utils import *


FLANKS = list(product("ATCG", repeat=2))


def plot_statistics(statistics, destination):
    """
    Desc:
        Plot mutation-spectrum statistics in a fixed category order.
    Args:
        statistics, Value supplied for statistics.
        destination, Figure output directory.
    """

    required = {"dataset", "group", "signature", "context", "count"}
    if not required.issubset(statistics.columns):
        raise ValueError(
            f"The statistics table is missing columns：{required - set(statistics.columns)}"
        )
    statistics = statistics.copy()
    statistics["count"] = pd.to_numeric(statistics["count"], errors="raise")
    if (
        not np.isfinite(statistics["count"]).all()
        or (statistics["count"] < 0).any()
        or (statistics["count"] % 1 != 0).any()
    ):
        raise ValueError("count must contain non-negative integers")
    expected = pd.MultiIndex.from_tuples(
        [
            (d, g, s, a + s[0] + b)
            for d in ("noncoding", "synonymous")
            for g in GROUPS
            for s in SIGNATURES
            for a, b in FLANKS
        ],
        names=["dataset", "group", "signature", "context"],
    )
    actual = pd.MultiIndex.from_frame(statistics[list(expected.names)])
    if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError(
            "The table must contain unique complete 96-channel records for both datasets and all groups"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        fig, axes = plt.subplots(2, 6, figsize=(23, 8), sharex=True, sharey="row")
        for row_id, dataset in enumerate(("noncoding", "synonymous")):
            subset = statistics[(statistics.dataset == dataset) & (statistics.group == group)]

            total = subset["count"].sum()
            counter = subset.set_index(["signature", "context"])["count"].to_dict()
            for col, signature in enumerate(SIGNATURES):
                contexts = [left + signature[0] + right for left, right in FLANKS]
                counts = [counter.get((signature, tri), 0) for tri in contexts]
                ax = axes[row_id, col]
                contributions = np.asarray(counts, dtype=float) / total if total else np.zeros(16)
                ax.bar(
                    np.arange(16),
                    contributions,
                    color=COLORS[col],
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_xticks(
                    np.arange(16), [f"{a}·{b}" for a, b in FLANKS], rotation=90, fontsize=8
                )
                if row_id == 0:
                    ax.add_patch(
                        Rectangle(
                            (0, 1),
                            1,
                            0.14,
                            transform=ax.transAxes,
                            facecolor="#eeeeee",
                            edgecolor="black",
                            linewidth=0.8,
                            clip_on=False,
                        )
                    )
                    ax.text(
                        0.5,
                        1.07,
                        signature,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                if col == 5:
                    label = "Noncoding variants" if row_id == 0 else "Synonymous variants"
                    ax.text(
                        1.08,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        rotation=270,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.8)
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=0.15)
                ax.set_axisbelow(True)
        fig.supylabel("Relative contribution")
        fig.supxlabel("context")
        fig.tight_layout(rect=(0.02, 0.03, 0.97, 0.98))
        save_figure(fig, destination, f"{group}_tri_96")


def plot_from_file(statistics_path, destination):
    """
    Desc:
        Execute the plot from file operation.
    Args:
        statistics_path, Value supplied for statistics_path.
        destination, Figure output directory.
    """
    plot_statistics(pd.read_csv(statistics_path, sep="\t"), destination)
