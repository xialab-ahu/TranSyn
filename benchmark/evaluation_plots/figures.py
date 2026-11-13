"""
@description: Plot precomputed benchmark results without recalculating statistics.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from .style import DARK, BLUE, COLORS, DISPLAY_METHODS, display_name, save

def pairwise_plot(results, path, relative=False):
    """
    Desc:
        Plot a pairwise common-set bar chart for one primary metric with paired difference intervals and white hatching. Display only synVep(mean), abbreviated as synVep.

    Args:
        results (pandas.DataFrame): Pairwise result table for one metric: AUROC for Test I and AP for Test II.
        path (pathlib.Path or str): Complete PNG output path.
        relative (bool): If True, plot relative improvement percentages; if False, plot absolute differences in percentage points.

    Returns:
        None: Saves the figure without modifying the input table.
    """
    results = results[results.tool.isin(DISPLAY_METHODS)].copy()
    tools = results.tool.tolist()
    color = '#253356' if results.metric.iloc[0] == 'AUROC' else '#82c2cc'
    fig, ax = plt.subplots(figsize=(8, 5))
    value = results.relative_improvement_percent if relative else 100*results.delta
    low = results.relative_ci_low if relative else 100*results.delta_ci_low
    high = results.relative_ci_high if relative else 100*results.delta_ci_high
    y = np.arange(len(tools))
    ax.barh(y, value, height=.6, color=color, edgecolor='white',
            hatch='////')
    for yy, lo, hi in zip(y, low, high):
        ax.plot([lo, hi], [yy, yy], color='#444444', lw=.9)
        ax.plot([lo,lo,np.nan,hi,hi], [yy-.06,yy+.06,np.nan,yy-.06,yy+.06], color='#444444', lw=.9)
    ax.set_yticks(y, [])
    positions = [(-.24, 'left'), (-.085, 'right'), (-.025, 'right')]
    for i, row in enumerate(results.itertuples()):
        for text, (x, align) in zip([display_name(row.tool), str(row.n), str(row.positive)], positions):
            ax.text(x, i, text, transform=ax.get_yaxis_transform(), ha=align, va='center', color=DARK, fontsize=11.5)
    for text, (x, align) in zip(['VEP','N','P'], positions):
        ax.text(x, 1.025, text, transform=ax.transAxes, ha=align, color=DARK, fontweight='bold', fontsize=13)
    ax.invert_yaxis()
    ax.axvline(0, color='#555555', lw=.7)
    name = 'AUC' if results.metric.iloc[0] == 'AUROC' else 'AUPR'
    ax.set_xlabel(f'{name}: TranSyn relative improvement (%)' if relative else f'{name}: TranSyn − comparator (percentage points)', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(axis='x', alpha=.15)
    save(fig, path, hide_axes=False)


def curve_plot(curves, table, dataset, path):
    """
    Desc:
        Plot ROC or PR curves on the unified common set. Use dark/light blue solid lines for the top two methods and gray dashed lines with distinct markers for the remaining methods. Show metrics and 95% intervals in the legend and retain 0.03 margins on both axes.

    Args:
        curves (pandas.DataFrame): Curve-coordinate table from the statistics package containing tool, x, and y.
        table (pandas.DataFrame): Metric and interval table returned by common_evaluate.
        dataset (str): I plots ROC and displays AUC; II plots PR and displays AUPR defined as AP.
        path (pathlib.Path or str): Complete PNG output path.

    Returns:
        None: Saves the curve figure and displays only synVep represented by its mean score.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    metric = 'AUROC' if dataset == 'I' else 'AP'
    rows = table[table.metric.eq(metric) & table.tool.isin(DISPLAY_METHODS)].set_index('tool')
    ranked = rows.sort_values('estimate', ascending=False, kind='stable').index.tolist()
    top = ranked[:2]
    highlights = dict(zip(top, ['#103057', '#92ccd7']))
    markers = ['o', 's', '^', 'D', '>', 'P', '*', 'X', 'p', 'h']
    marker_map = dict(zip(DISPLAY_METHODS, markers))
    for tool in ranked:
        row = rows.loc[tool]
        points = curves[curves.tool.eq(tool)]
        xx, yy = points.x.to_numpy(), points.y.to_numpy()
        color = highlights.get(tool, '#AAAAAA')
        name = 'AUC' if dataset == 'I' else 'AUPR'
        label = f'{display_name(tool)} ({name}={row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}])'
        options = dict(color=color, linestyle='-' if tool in top else '--',
                       linewidth=1.5 if tool in top else 1, label=label)
        if tool not in top:
            options.update(marker=marker_map[tool], markevery=.1, markersize=7,
                           markerfacecolor=to_rgba(color, alpha=.3), markeredgecolor='#333333')
        ax.plot(xx, yy, **options)
    # Random baseline: the ROC diagonal or the positive proportion in the current unified common set for PR.
    if dataset == 'I':
        ax.plot([0, 1], [0, 1], '--', color='#666666', linewidth=1,
                label='Random guess (AUC=0.500)', zorder=0)
    else:
        prevalence = float(rows.iloc[0]['positive']/rows.iloc[0]['n'])
        ax.axhline(prevalence, linestyle='--', color='#666666', linewidth=1,
                   label=f'Random guess (prevalence={prevalence:.3f})', zorder=0)
    ax.set_xlabel('False positive rate' if dataset == 'I' else 'Recall', fontsize=15)
    ax.set_ylabel('True positive rate' if dataset == 'I' else 'Precision', fontsize=15)
    ax.set(xlim=(-.03,1.03), ylim=(-.03,1.03))
    ax.set_xticks(np.arange(0,1.1,.1))
    ax.set_yticks(np.arange(0,1.1,.1))
    ax.tick_params(labelsize=15)
    # ax.legend(frameon=False, fontsize=13, loc='lower right' if dataset=='I' else 'lower left')
    ax.legend(frameon=True, facecolor='white', framealpha=0.6, fontsize=13, loc='lower left' if dataset=='II' else 'lower right')
    save(fig, path)


def spearman_plot(full_corr, path):
    """
    Desc:
        Plot the complete precomputed Spearman matrix from the statistics package, displaying only synVep represented by its mean score.
    Args:
        full_corr (pandas.DataFrame): Correlation matrix containing all statistical score columns.
        path (pathlib.Path or str): PNG output path.
    Returns:
        None: Saves only the PNG without calculating or modifying statistical results.
    """
    corr = full_corr.loc[DISPLAY_METHODS, DISPLAY_METHODS]
    names = [display_name(m) for m in DISPLAY_METHODS]
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#103057', '#F5EFEA']) # '#103057', '#F5EFEA'
    im = ax.imshow(corr, cmap=cmap, vmin=float(np.nanmin(corr.to_numpy())), vmax=1)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha='right', fontsize=14)
    ax.set_yticks(np.arange(len(names)), names, fontsize=14)
    ax.set_xticks(np.arange(len(names)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(names)+1)-.5, minor=True)
    ax.grid(which='minor', color='#dddddd', linewidth=.5)
    ax.set_aspect('equal')
    for i in range(len(names)):
        for j in range(len(names)):
            value = corr.iloc[i, j]
            rgba = im.cmap(im.norm(value))
            luminance = np.dot(rgba[:3], [.299, .587, .114])
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=12,
                    color='white' if luminance < .5 else 'black')
    bar=fig.colorbar(im,ax=ax,shrink=.7,aspect=30,pad=.02)
    bar.set_label('Spearman correlation',fontsize=14)
    bar.ax.tick_params(labelsize=13)
    save(fig,path)


def hsr_plot(table, path):
    """
    Desc:
        Display the precomputed top three methods by 95% hsr-AUC, with tools on the x-axis, spacing between bars, and no legend.
        Assign three colors by rank; error bars show the existing pointwise 95% bootstrap intervals.
    Args:
        table (pandas.DataFrame): Statistical results containing tool, metric, estimate, ci_low, and ci_high.
        path (pathlib.Path or str): PNG output path.
    Returns:
        None: Plots only, without updating ranking tables or rerunning bootstrap.
    """
    top = (table[table.metric.eq('hsr_AUC_95') & table.tool.isin(DISPLAY_METHODS)]
           .sort_values('estimate', ascending=False, kind='stable').head(3))
    if top.empty:
        raise ValueError('No hsr_AUC_95 results available for plotting')
    colors = ['#253356', '#82c2cc', '#c2c3c4']
    positions = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.bar(positions, top.estimate, width=.6, color=colors[:len(top)])
    for x, row in zip(positions, top.itertuples()):
        # Draw intervals from their endpoints to support percentile intervals that do not contain the point estimate.
        ax.vlines(x, row.ci_low, row.ci_high, color='#444444', lw=.9)
        ax.hlines([row.ci_low, row.ci_high], x-.06, x+.06, color='#444444', lw=.9)
        ax.text(x, max(row.estimate, row.ci_high)+.02, f'{row.estimate:.3f}',
                ha='center', va='bottom', fontsize=16)
    ax.set_xticks(positions, [display_name(t) for t in top.tool])
    # ax.set_xlabel('Tool', fontsize=15)
    ax.set_ylabel('Normalized hsr-AUC (TPR ≥ 95%)', fontsize=15)
    ax.set(ylim=(0, 1.05), xlim=(-.6, len(top)-.4))
    ax.set_yticks(np.arange(0, 1.1, .1))
    ax.tick_params(labelsize=15)
    save(fig, path)

def recall_plot(rows, path):
    """
    Desc:
        Read the precomputed Recall table and plot the 1%-20% curve with pointwise 95% confidence-interval error bars.
    Args:
        rows (pandas.DataFrame): Statistics table containing tool, fraction, estimate, ci_low, and ci_high.
        path (pathlib.Path or str): PNG output path.
    Returns:
        None: Plots only, without reranking samples, calculating metrics, or writing statistical tables.
    """
    fig,ax=plt.subplots(figsize=(6.2,5.5))
    for tool in DISPLAY_METHODS:
        row=rows[rows.tool.eq(tool)].sort_values('fraction')
        x=row.fraction.to_numpy()
        color=COLORS[tool]
        # Plot confidence-interval endpoints directly to support percentile intervals that do not contain the point estimate.
        for xx,lo,hi in zip(x,row.ci_low,row.ci_high):
            ax.plot([xx,xx],[lo,hi],color=color,lw=.55,alpha=.45,zorder=1)
            ax.plot([xx-.0006,xx+.0006,np.nan,xx-.0006,xx+.0006],
                    [lo,lo,np.nan,hi,hi],color=color,lw=.55,alpha=.45,zorder=1)
        ax.plot(x,row.estimate,color=color,lw=1,marker='o',markersize=8,
                markerfacecolor=to_rgba(color,alpha=.5),markeredgecolor=to_rgba(color,alpha=.5),
                zorder=10 if tool=='TranSyn' else 5,label=display_name(tool))
    ax.set_xlabel('Top fraction (%)',fontsize=15)
    ax.set_ylabel('Recall',fontsize=15)
    ax.set_yticks(np.arange(0,1.1,.1))
    ax.set_xticks(np.arange(0,.22,.02),np.arange(0,22,2))
    ax.set(ylim=(-.05,1.05),xlim=(-.005,.22))
    ax.tick_params(labelsize=15)
    ax.legend(frameon=True, framealpha=0.3, fontsize=12, ncol=2,loc='lower right')
    save(fig,path)

