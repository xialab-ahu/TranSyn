import argparse
from pathlib import Path
import pandas as pd
from evaluation_config import DEFAULT_OUTPUT
from .style import configure_style
from .figures import pairwise_plot, curve_plot, spearman_plot, hsr_plot, recall_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset', choices=['I', 'II', 'all'], default='all')
    args = parser.parse_args()
    configure_style()
    for dataset in ['I', 'II']:
        if args.dataset != 'all' and args.dataset != dataset:
            continue
        out = args.input/f'Test.set.{dataset}'
        pair = pd.read_csv(out/'pairwise.metrics.tsv', sep='\t')
        table = pd.read_csv(out/'common.metrics.tsv', sep='\t')
        curves = pd.read_csv(out/'common.curve.tsv', sep='\t')
        for relative, suffix in [(False, 'delta'), (True, 'relative')]:
            pairwise_plot(pair, out/f'pairwise.{suffix}.png', relative)
        curve_plot(curves, table, dataset, out/'common.curve.png')
        if dataset == 'I':
            corr = pd.read_csv(out/'common.spearman.tsv', sep='\t', index_col=0)
            spearman_plot(corr, out/'common.spearman.png')
            hsr_plot(table, out/'common.hsr_top3.png')
        else:
            recall = pd.read_csv(out/'common.recall.tsv', sep='\t')
            recall_plot(recall, out/'common.recall.png')
        print(f'Test {dataset}: PNG figures were updated; statistical tables were not modified.', flush=True)


if __name__ == '__main__':
    main()
