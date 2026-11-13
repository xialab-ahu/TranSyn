"""
@description: visualize the result of bootstrap
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
COLORS={'Test.set.I':'#253356', 'Test.set.II':'#82c2cc'}


def save(fig,  path):
    for ax in fig.axes:
        for spine in ax.spines.values():spine.set_linewidth(.5)
        ax.tick_params(width=.5)
    fig.tight_layout();fig.savefig(path, dpi=250, bbox_inches='tight');plt.close(fig)


def reliability_plot(table, name, path):
    fig, (ax, counts) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
    valid = table.n > 0
    r = table.loc[valid]
    color = COLORS[name]
    
    ax.plot([0, 1], [0, 1], '--', color='#777777', label='Ideal calibration', lw=1)
    ax.plot(r.mean_score, r.observed_fraction, 'o-', color=color, markersize=6, label='TranSyn')
    for row in r.itertuples():
        if np.isfinite([row.ci_low, row.ci_high]).all():
            ax.vlines(row.mean_score, row.ci_low, row.ci_high, color=color, lw=.9)
            ax.hlines([row.ci_low, row.ci_high], row.mean_score-.012, row.mean_score+.012, color=color, lw=.9)
        offsets = {1:(12, 12),  2:(12, 14),  3:(-12, -26),  4:(12, -26),  5:(-16, 16)}
        offset = offsets.get(int(row.bin), (12, 12))
        ax.annotate(f'n={int(row.n)}',  (row.mean_score, row.observed_fraction), 
                    xytext=offset, textcoords='offset points', fontsize=12,  fontweight='bold', 
                    ha='right' if offset[0]<0 else 'left', va='center', zorder=10, 
                    bbox=dict(boxstyle='round, pad=.25', facecolor='white', edgecolor='none', alpha=.5), 
                    arrowprops=dict(arrowstyle='-', color=color, lw=.7, alpha=.7))
    ax.set(ylabel='Observed pathogenic fraction', xlim=(-.03, 1.03), ylim=(-.03, 1.13))
    ax.legend(frameon=False)
    counts.bar((table.left+table.right)/2, table.n, width=(table.right-table.left)*.8, color=color)
    counts.set(xlabel='Mean predicted score / score bin', ylabel='Variants', xticks=np.linspace(0, 1, 6))
    save(fig, path)


def metrics_plot(tables, path):
    fig, axes=plt.subplots(1, 2, figsize=(6, 4.5))
    for ax, metric in zip(axes, ['Brier', 'ECE']):
        upper = max(t.set_index('metric').loc[metric, 'ci_high'] for t in tables.values())
        if metric == 'Brier':
            upper = max(upper, max(t.set_index('metric').loc['Constant_reference_Brier', 'estimate'] for t in tables.values()))
        label_gap = upper*.04
        for i, (name, table) in enumerate(tables.items()):
            row=table.set_index('metric').loc[metric]
            ax.bar(i, row.estimate, width=.55, color=COLORS[name])
            ax.vlines(i, row.ci_low, row.ci_high, color='#444444', lw=1)
            ax.hlines([row.ci_low, row.ci_high], i-.06, i+.06, color='#444444', lw=1)
            if metric=='Brier':
                ref=table.set_index('metric').loc['Constant_reference_Brier', 'estimate']
                ax.plot(i, ref, '_', markersize=20, color='#e70f1e', label='Constant reference' if i==0 else None)
            ax.text(i, row.ci_high+label_gap, f'{row.estimate:.3f}', ha='center', va='bottom', fontsize=12)
        ax.set_xticks([0, 1], ['Test I', 'Test II']);ax.set_ylabel(metric);ax.set_ylim(0, upper*1.30)
        if metric=='Brier':ax.legend(frameon=True, fontsize=12)
    save(fig, path)


def prevalence_plot(tables, path, threshold):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for index, (name, table) in enumerate(tables.items()):
        color = COLORS[name]
        ax.plot(table.prevalence*100, table.ppv, color=color, alpha=.75, 
                linestyle='-' if index == 0 else '--', marker='o' if index==0 else 's', 
                markersize=7, markerfacecolor=color if index==0 else 'white', 
                label=name.replace('Test.set.', 'Test '))
        for row in table.itertuples():
            ax.vlines(row.prevalence*100, row.ci_low, row.ci_high, color=color, lw=.9, alpha = .4)
    ax.set_xscale('log')
    ax.set(xlabel='Assumed pathogenic-variant proportion (%)', ylabel='Projected precision (PPV)', ylim = (-.03, 1.03))
    ax.set_xticks(tables['Test.set.I'].prevalence*100, [f'{x*100:g}' for x in tables['Test.set.I'].prevalence])
    ax.legend(frameon=False, title=f'Reference threshold > {threshold:g}')
    save(fig, path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'results');args=parser.parse_args()
    for font in (HERE.parent/'assets/fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(font))
    font_manager.findfont(font_manager.FontProperties(family='Arial'), fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial', 'font.size':11})

    meta=json.loads((args.input/'metadata.json').read_text())
    summaries={}
    scenarios={}
    for name in ['Test.set.I', 'Test.set.II']:
        summaries[name]=pd.read_csv(args.input/f'{name}.summary.tsv', sep='\t')
        scenarios[name]=pd.read_csv(args.input/f'{name}.prevalence.tsv', sep='\t')
        bins=pd.read_csv(args.input/f'{name}.reliability.tsv', sep='\t')
        reliability_plot(bins, name, args.input/f'{name}.reliability.png')
    metrics_plot(summaries, args.input/'calibration_metrics.png')
    prevalence_plot(scenarios, args.input/'prevalence_scenarios.png', meta['threshold'])

if __name__=='__main__':main()
