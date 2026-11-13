"""
@description: Export auxiliary benchmark statistics and curve tables.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from evaluation_config import METHODS


def export_auxiliary(sub, table, dataset, out, overwrite=True):
    """
    Desc:
        Export ROC/PR coordinates, the Spearman matrix, and the complete Top-3 or Recall table.
    Args:
        sub (pandas.DataFrame): Fixed unified common set containing labels and all scores.
        table (pandas.DataFrame): Main statistics table containing metrics and confidence intervals.
        dataset (str): I or II.
        out (pathlib.Path): Output directory.
        overwrite (bool): Whether to overwrite existing auxiliary tables; use False when migrating existing results.
    """
    curves = []
    for tool in METHODS:
        if dataset == 'I':
            x, y, _ = roc_curve(sub.label, sub[tool], drop_intermediate=False)
        else:
            precision, recall, _ = precision_recall_curve(sub.label, sub[tool])
            x, y = recall, precision
        curves.append(pd.DataFrame({'tool': tool, 'x': x, 'y': y}))
    frames = {'common.curve.tsv': pd.concat(curves, ignore_index=True)}
    if dataset == 'I':
        corr = sub[METHODS].corr(method='spearman')
        target = out/'common.spearman.tsv'
        if overwrite or not target.exists():
            corr.to_csv(target, sep='\t')
        frames['common.hsr_top3.tsv'] = (table[table.metric.eq('hsr_AUC_95')]
            .sort_values('estimate', ascending=False, kind='stable').head(3))
    else:
        recall = table[table.metric.str.startswith('Recall@')].copy()
        recall['fraction'] = recall.metric.str.split('@').str[-1].astype(float)
        recall['k'] = [max(1, int(len(sub)*p)) for p in recall.fraction]
        frames['common.recall.tsv'] = recall
    for name, frame in frames.items():
        if overwrite or not (out/name).exists():
            frame.to_csv(out/name, sep='\t', index=False)
