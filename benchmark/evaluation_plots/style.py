"""
@description: Configure consistent styles and output settings for benchmark figures.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from evaluation_config import ROOT, METHODS

DARK, BLUE = '#093e6a', '#59a3cf'
COLORS = {'TranSyn': '#e70f1e', 'seDSM': '#f77460', 'TraP': '#fcb4a6',
          'SilVA': '#a5abbb', 'CADD': '#103057', 'CAPICE': '#92ccd7',
          'usDSM': '#b4c7d5', 'SynScore': '#d7b8b4',
          'synVep(mean)': '#cccccc', 'synVep(max)': '#cccccc'}
DISPLAY_METHODS = [m for m in METHODS if m != 'synVep(max)']



def display_name(tool):
    return 'synVep' if tool == 'synVep(mean)' else tool


def configure_style():
    for path in (ROOT/'assets/fonts').glob('*.ttf'):
        font_manager.fontManager.addfont(str(path))
    for weight in ['normal', 'bold']:
        font_manager.findfont(font_manager.FontProperties(family='Arial', weight=weight), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})


def clean_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, bottom=False, top=False, left=False, right=False)


def save(fig, path, hide_axes=False):
    """
    Desc:
        Apply a consistent layout and save PNG only. Pairwise plots hide axes spines and tick lines; other figures retain thin spines and major tick lines.

    Args:
        fig (matplotlib.figure.Figure): Figure object to save.
        path (pathlib.Path or str): Complete PNG output path; the parent directory must already exist.
        hide_axes (bool): Whether to call clean_axes to hide spines and tick lines for all subplots; default is False.
    """
    for ax in fig.axes:
        if hide_axes:
            clean_axes(ax)
        else:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.5)
            ax.tick_params(axis='both', which='major', length=3.5, bottom=True, left=True,
                           top=False, right=False, width=.5)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)

