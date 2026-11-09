"""
@description: provide shared mutation-spectrum utilities
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import os
from pathlib import Path
import csv
import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TRANSYN_DATA_ROOT", ".")).resolve()
PROJECT = Path(os.environ.get("TRANSYN_PROJECT_DATA", ROOT / "transfer_learning_project")).resolve()


SIGNATURES = ["C>T", "C>A", "C>G", "T>A", "T>C", "T>G"]

# COLORS = ["#df4545", "#40b5d5", "#333333", "#aaaaaa", "#94b942", "#e5b4bc"]
# COLORS = ["#c3185c", "#1a9894", "#e7811f", "#d6a019", "#9468bd", "#2165a9"]
# COLORS = ["#ea3a47", "#feadb6", "#9a9a9a", "#6396f2", "#feca04", "#010101"]
COLORS = ["#8f0e20", "#159eb0", "#e56e17", "#926399", "#75a866", "#cda57f"]


COMP = str.maketrans("ACGT", "TGCA")

GROUPS = ["benign", "pathogenic", "combined"]

plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
)


def setup(name):
    """
    Desc:
        Create the output directory and configure a task-specific logger.
    Args:
        name, Output or task name.
    """
    destination = BASE / "results" / name
    destination.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(destination / "invalid_records.log", mode="w")
    handler.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
    logger.addHandler(handler)
    return destination, logger


def rows(path):
    """
    Desc:
        Yield each row from a tab-separated input file.
    Args:
        path, Input filesystem path.
    """
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            yield reader.line_num, record


def canonical(ref, alt):
    """
    Desc:
        Convert a nucleotide substitution to the pyrimidine-oriented signature.
    Args:
        ref, Reference allele.
        alt, Alternate allele.
    """
    ref, alt = ref.upper(), alt.upper()
    if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
        raise ValueError("invalid_SNV")
    reverse = ref in "AG"
    if reverse:
        ref, alt = ref.translate(COMP), alt.translate(COMP)
    return f"{ref}>{alt}", reverse


def save_figure(fig, destination, name):
    """
    Desc:
        Save a figure in the supported publication formats.
    Args:
        fig, Value supplied for fig.
        destination, Figure output directory.
        name, Output or task name.
    """
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.show()
    # plt.close(fig)
