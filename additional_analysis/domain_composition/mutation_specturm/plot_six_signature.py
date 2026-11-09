"""
@description: plot six-class SNV spectra
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from signature_utils import *
from matplotlib.ticker import FuncFormatter

BINS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]


def plot_statistics(statistics, destination):
    """
    Desc:
        Plot mutation-spectrum statistics in a fixed category order.
    Args:
        statistics, Value supplied for statistics.
        destination, Figure output directory.
    """
    required = {
        "dataset",
        "group",
        "signature",
        "cds_bin",
        "count",
        "signature_total",
        "deviation_percentage_points",
    }
    if not required.issubset(statistics.columns):
        raise ValueError(
            f"The statistics table is missing columns：{required - set(statistics.columns)}"
        )
    expected = pd.MultiIndex.from_product(
        [["missense", "synonymous"], GROUPS, SIGNATURES, BINS],
        names=["dataset", "group", "signature", "cds_bin"],
    )
    actual = pd.MultiIndex.from_frame(statistics[list(expected.names)])
    if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError(
            "The table must contain all five bins and six substitutions for both datasets and all groups"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for dataset in ("missense", "synonymous"):
        for group in GROUPS:
            subset = statistics[(statistics.dataset == dataset) & (statistics.group == group)]
            fig, ax = plt.subplots(figsize=(11, 5))
            for i, signature in enumerate(SIGNATURES):
                values = subset[subset.signature == signature].set_index("cds_bin").loc[BINS]
                ax.bar(
                    np.arange(5) + (i - 2.5) * 0.125,
                    values.deviation_percentage_points.to_numpy(),
                    width=0.12,
                    color=COLORS[i],
                    edgecolor="black",
                    linewidth=0.5,
                    label=signature,
                    hatch="///" if group == "pathogenic" else None,
                )
            ax.axhline(0, color="black", linewidth=1)
            ax.set(
                xticks=np.arange(5),
                xticklabels=BINS,
                xlabel="Location within CDS (5′ to 3′)",
                ylabel="Distribution across coding sequence length",
            )

            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value + 20:.1f}%"))
            ax.legend(ncol=3, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            save_figure(fig, destination, f"{dataset}_{group}_cds_six")


def plot_from_file(statistics_path, destination):
    """
    Desc:
        Execute the plot from file operation.
    Args:
        statistics_path, Value supplied for statistics_path.
        destination, Figure output directory.
    """
    plot_statistics(pd.read_csv(statistics_path, sep="\t"), destination)
