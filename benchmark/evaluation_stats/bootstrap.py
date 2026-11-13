"""
@description: Perform paired bootstrap comparisons using gene, transcript, or variant resampling units.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import numpy as np
import pandas as pd
from evaluation_config import METHODS, FRACTIONS
from .metrics import metrics, measure

def evaluate(df, repetitions, seed, bootstrap_unit="transcript_id"):
    """
    Desc:
        For TranSyn and each comparator, identify their common set of finite scores, perform paired bootstrap resampling, and report coverage and metric intervals. Gene and transcript modes resample whole clusters; variant mode uses class-stratified resampling.

    Args:
        df, input table containing variant38, label, TranSyn, comparator scores, and the required cluster annotations; variant38 must be unique.
        repetitions, number of valid bootstrap replicates.
        seed, random seed; the random-number generator is reset independently for each comparator.
        bootstrap_unit, gene_id, transcript_id, or variant; the default is transcript_id. Transcript mode uses synVep, whereas the other modes use mean/max scores.
    """
    # 1 🚩 Validate basic information
    # Check for duplicate variant identifiers
    if df.variant38.duplicated().any():
        raise ValueError('Duplicate variant38 identifiers: resolve before evaluation.')
    # Check that labels are non-missing binary values encoded as 0 or 1
    if df.label.isna().any() or not set(df.label.unique()) <= {0, 1}:
        raise ValueError('Labels must be nonmissing binary 0/1.')
    # Exclude metadata columns; all remaining columns represent tools
    tools = [c for c in df if c not in ['variant38', 'label', 'TranSyn', 'transcript_id', 'gene_id', 'gene_symbol']]
    # Select synVep scores according to the bootstrap unit
    if bootstrap_unit == 'transcript_id':
        # For transcript-level resampling, exclude mean/max aggregate scores
        tools = [c for c in tools if c not in ['synVep(max)', 'synVep(mean)']]
        if 'synVep' not in tools:
            raise ValueError('Transcript bootstrap requires matched synVep scores')
    else:
        # For gene- or variant-level resampling, exclude transcript-level scores
        tools = [c for c in tools if c != 'synVep']

    # 2 🚩 Store coverage statistics
    coverage, results, memberships = [], [], []
    # Calculate metrics for TranSyn paired with each comparator
    for tool in ['TranSyn'] + tools:
        # Require numeric values and raise an error for non-numeric entries
        scores = pd.to_numeric(df[tool], errors='raise')
        # Exclude NaN and infinite values
        valid = np.isfinite(scores)
        row = dict(tool=tool,
                   total=len(df),
                   scored=int(valid.sum()),
                   missing=int((~valid).sum()),
                   coverage=float(valid.mean()))
        # Summarize missingness separately by class to identify class-dependent missingness
        for label, name in [(1, 'positive'), (0, 'negative')]:
            mask = df.label.eq(label)
            row[name + '_total'] = int(mask.sum())
            row[name + '_missing'] = int((mask & ~valid).sum())
        coverage.append(row)
    
    # 3 🚩 Cluster annotations
    # Non-variant modes require a complete cluster column (gene ID or transcript ID); otherwise whole-cluster resampling is impossible
    if bootstrap_unit != "variant" and (bootstrap_unit not in df or df[bootstrap_unit].isna().any()):
        raise ValueError(f"Missing cluster annotations: {bootstrap_unit}")
    # Process each tool
    for tool in tools:
        rng = np.random.default_rng(seed) # Reset the RNG independently for each comparator for reproducibility and independence between comparisons
        scores = df[['TranSyn', tool]].apply(pd.to_numeric, errors='raise')  # Select the TranSyn and comparator score columns
        valid = np.isfinite(scores).all(axis=1) # Common valid set
        sub = df.loc[valid]
        y = sub.label.to_numpy(dtype=int) # 
        # Positions of positive and negative classes within sub; AUROC/AP is undefined if either class is absent
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if not len(positive) or not len(negative):
            raise ValueError(f'{tool}: paired subset lacks one class.')
        
        a = scores.loc[valid, 'TranSyn'].to_numpy()
        # Reverse GPN-MSA scores; higher scores indicate greater pathogenicity for all other tools
        b = scores.loc[valid, tool].to_numpy() * (-1 if tool == 'GPN-MSA' else 1)
        point = np.stack([metrics(y, a), metrics(y, b)])
        # Store model-by-metric results for each bootstrap replicate
        draws = np.empty((repetitions, 2, 2))
        # Construct resampling units:
        # - gene/transcript modes: resample whole clusters to preserve within-cluster dependence
        # - variant mode: use None and apply stratified resampling below
        clusters = ([np.asarray(v) for v in sub.reset_index(drop=True).groupby(bootstrap_unit, sort=False).indices.values()]
                    if bootstrap_unit != 'variant' else None)
        
        # Whole clusters preserve within-gene or within-transcript dependence; resample single-class draws without splitting clusters.
        attempts = 0
        for i in range(repetitions):
            while True:
                attempts += 1
                # Prevent infinite resampling under extreme class imbalance
                if attempts > repetitions * 100:
                    raise ValueError(f'{tool}: too many single-class bootstrap draws')
                if clusters is None:
                    # variant mode: resample each class independently with replacement while preserving its sample count
                    idx = np.concatenate([rng.choice(positive, len(positive), replace=True),
                                          rng.choice(negative, len(negative), replace=True)])
                else:
                    # cluster mode: resample whole clusters with replacement without splitting samples within a cluster
                    idx = np.concatenate([clusters[k] for k in rng.integers(len(clusters), size=len(clusters))])
                # Reject single-class samples so that metrics remain defined
                if np.unique(y[idx]).size == 2:
                    break
            # Use exactly the same indices for both models; do not bootstrap them independently.
            draws[i] = np.stack([metrics(y[idx], a[idx]), metrics(y[idx], b[idx])])
        
        # Record common-set members for auditing and traceability
        for _, variant in sub.iterrows():
            memberships.append(dict(tool=tool, variant38=variant.variant38, label=int(variant.label),
                                    transcript_id=variant.transcript_id, gene_id=variant.get('gene_id', '')))
        
        # 4 🚩 Summarize metrics and intervals
        for j, metric in enumerate(['AUROC', 'AP']):
            # Difference: TranSyn minus comparator, calculated within each paired bootstrap replicate
            delta = draws[:, 0, j] - draws[:, 1, j]
            # Relative improvement is undefined when the comparator metric is zero; retain NaN and record the valid replicate count.
            relative = np.divide(100 * delta, draws[:, 1, j],
                                 out=np.full(repetitions, np.nan), where=draws[:, 1, j] != 0)
            row = dict(tool=tool, metric=metric, n=len(sub), positive=len(positive),
                       negative=len(negative), prevalence=float(y.mean()),
                       transyn=point[0, j], comparator=point[1, j],
                       delta=point[0, j]-point[1, j],
                       relative_improvement_percent=(100*(point[0,j]-point[1,j])/point[1,j]
                                                     if point[1,j] else np.nan),
                       bootstrap_repetitions=repetitions, seed=seed, bootstrap_unit=bootstrap_unit,
                       # In variant mode, clusters is None and the cluster count equals the sample count
                       n_clusters=(len(clusters) if clusters is not None else len(sub)),
                       # Number of rejected single-class samples, reflecting pressure from class imbalance
                       rejected_single_class_draws=attempts-repetitions,
                       # Number of valid replicates contributing to the relative interval
                       relative_valid_repetitions=int(np.isfinite(relative).sum()))
            # Calculate pointwise 95% intervals for the four statistics after excluding NaN values
            for key, values in [('transyn', draws[:,0,j]), ('comparator', draws[:,1,j]),
                                ('delta', delta), ('relative', relative)]:
                finite = values[np.isfinite(values)]
                lo, hi = np.quantile(finite, [.025,.975]) if len(finite) else (np.nan,np.nan)
                row[key+'_ci_low'], row[key+'_ci_high'] = lo, hi
            results.append(row)
        print(f'  {tool}: n={len(sub)}, positives={len(positive)}', flush=True)
    return pd.DataFrame(coverage), pd.DataFrame(results), pd.DataFrame(memberships)


def common_evaluate(df, dataset, repetitions, seed):
    """
    Desc:
        First fix a unified common set on which all specified scores are finite, then perform paired whole-gene bootstrap resampling. Every tool and metric uses the same sampled indices in each replicate; the common set is not reselected and models are not retrained.
    Args:
        df (pandas.DataFrame): Table containing the scores specified by METHODS and columns including variant38, label, and gene_id.
        dataset (str): I or II, which determines the metric set.
        repetitions (int): Number of valid gene-level bootstrap replicates.
        seed (int): Random seed.
    """
    scores = df[METHODS].apply(pd.to_numeric, errors='raise')
    sub = df.loc[np.isfinite(scores).all(axis=1)].reset_index(drop=True)
    if sub.empty or sub.gene_id.isna().any() or sub.label.nunique() != 2:
        raise ValueError('Common set must have gene annotations and both classes')
    y = sub.label.to_numpy(int)
    matrix = sub[METHODS].to_numpy(float)
    labels = ['AUROC', 'hsr_AUC_95'] if dataset == 'I' else ['AP']+[f'Recall@{p:.2f}' for p in FRACTIONS]
    point = np.stack([measure(y, matrix[:, j], dataset) for j in range(len(METHODS))])
    draws = np.empty((repetitions, len(METHODS), len(labels)))
    groups = list(sub.groupby('gene_id', sort=False).indices.values())
    rng = np.random.default_rng(seed)
    attempts = 0
    # Share the sampled gene indices across all tools in each replicate to form paired comparisons.
    for i in range(repetitions):
        while True:
            attempts += 1
            if attempts > repetitions*100:
                raise ValueError('Too many single-class gene samples')
            idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            if np.unique(y[idx]).size == 2:
                break
        for j in range(len(METHODS)):
            draws[i, j] = measure(y[idx], matrix[idx, j], dataset)
    rows, differences = [], []
    for j, tool in enumerate(METHODS):
        for m, label in enumerate(labels):
            lo, hi = np.quantile(draws[:, j, m], [.025, .975])
            rows.append(dict(tool=tool, metric=label, estimate=point[j,m], ci_low=lo, ci_high=hi,
                             n=len(sub), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                             n_genes=len(groups), bootstrap_repetitions=repetitions, seed=seed,
                             rejected_single_class_draws=attempts-repetitions))
            if j:
                delta = draws[:, 0, m]-draws[:, j, m]
                lo, hi = np.quantile(delta, [.025, .975])
                differences.append(dict(tool=tool, metric=label, delta=point[0,m]-point[j,m],
                                        ci_low=lo, ci_high=hi))
    return sub, pd.DataFrame(rows), pd.DataFrame(differences)

