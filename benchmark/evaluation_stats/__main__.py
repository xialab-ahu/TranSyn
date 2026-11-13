import argparse
import hashlib
import json
import sys
from pathlib import Path
from importlib.metadata import version
import pandas as pd
from evaluation_config import ROOT, METHODS, FRACTIONS, DEFAULT_OUTPUT
from .bootstrap import evaluate, common_evaluate
from .tables import export_auxiliary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset', choices=['I', 'II', 'all'], default='all')
    parser.add_argument('--export-existing', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be >=100')
    for dataset, primary in [('I', 'AUROC'), ('II', 'AP')]:
        if args.dataset != 'all' and args.dataset != dataset:
            continue
        out = args.output/f'Test.set.{dataset}'
        out.mkdir(parents=True, exist_ok=True)
        if args.export_existing:
            sub = pd.read_csv(out/'common.memberships.tsv', sep='\t')
            table = pd.read_csv(out/'common.metrics.tsv', sep='\t')
            export_auxiliary(sub, table, dataset, out, overwrite=False)
            print(f'Test {dataset}: missing auxiliary tables were added without running bootstrap.', flush=True)
            continue
        source = ROOT/'origin_data'/f'Test.set.{dataset}.benchmark.txt'
        df = pd.read_csv(source, sep='\t')[['variant38', 'label', 'transcript_id', 'gene_id']+METHODS]
        if df.gene_id.isna().any():
            raise ValueError('Missing genes')
        coverage, pair, members = evaluate(df, args.bootstrap, args.seed, 'gene_id')
        pair = pair[pair.metric.eq(primary)].reset_index(drop=True)
        sub, table, diff = common_evaluate(df, dataset, args.bootstrap, args.seed)
        for name, frame in [('coverage', coverage), ('pairwise.metrics', pair),
                            ('pairwise.memberships', members), ('common.memberships', sub),
                            ('common.metrics', table), ('common.differences', diff)]:
            frame.to_csv(out/f'{name}.tsv', sep='\t', index=False)
        export_auxiliary(sub, table, dataset, out)
        metadata = dict(dataset=dataset, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        methods=METHODS, n=len(sub), positive=int(sub.label.sum()), n_genes=int(sub.gene_id.nunique()),
                        bootstrap_repetitions=args.bootstrap, seed=args.seed, unit='gene_id',
                        hsr_sensitivity=.95 if dataset=='I' else None,
                        recall_fractions=FRACTIONS.tolist() if dataset=='II' else None,
                        intervals='pointwise percentile 95%, unadjusted for multiplicity',
                        python_executable=sys.executable,
                        packages={p: version(p) for p in ['scikit-learn','scipy','numpy','pandas','matplotlib']})
        (out/'analysis.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(f'Test {dataset}: n={len(sub)}, genes={sub.gene_id.nunique()}; statistical evaluation completed.', flush=True)


if __name__ == '__main__':
    main()
