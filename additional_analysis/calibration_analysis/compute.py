"""
@description: Compute independent calibration statistics using raw TranSyn outputs, gene-level bootstrap resampling, a fixed reference threshold, and assumed PPV scenarios.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, recall_score, precision_score, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = ['Brier', 'ECE', 'Constant_reference_Brier', 'Brier_minus_reference',
           'Sensitivity', 'Specificity', 'Observed_precision']


def reliability(y, p, n_bins):
    """
    Desc:
        Use scikit-learn to calculate the observed positive fraction and mean
        predicted probability in equal-width bins, and SciPy to count the
        observations in each bin. ECE is the weighted absolute difference
        between these package outputs; no calibration-fitting algorithm is
        implemented separately.
    Args:
        y (numpy.ndarray): Binary labels encoded as 0 or 1.
        p (numpy.ndarray): Raw probability-like outputs in the range [0, 1].
        n_bins (int): Fixed number of equal-width bins; the CLI default is 5.
    """
    edges = np.linspace(0., 1., n_bins+1)
    # scikit-learn uses searchsorted and assigns boundary values to the left
    # bin. Adjust the SciPy boundaries to preserve the same behavior.
    count_edges = edges.copy()
    count_edges[1:-1] = np.nextafter(count_edges[1:-1], np.inf)
    counts = binned_statistic(p, p, statistic='count', bins=count_edges).statistic.astype(int)
    fraction, mean = calibration_curve(y, p, n_bins=n_bins, strategy='uniform', pos_label=1)
    valid = counts > 0
    if valid.sum() != len(mean) or counts.sum() != len(y):
        raise ValueError('Calibration/count bin alignment failed')
    observed = np.full(n_bins, np.nan); predicted = observed.copy()
    observed[valid], predicted[valid] = fraction, mean
    ece = float(np.sum(counts[valid]/len(y)*np.abs(fraction-mean)))
    return pd.DataFrame({'bin': np.arange(1,n_bins+1), 'left': edges[:-1], 'right': edges[1:], 'n': counts, 'mean_score': predicted, 'observed_fraction': observed}), ece


def measures(y, p, n_bins, threshold, reference_probability):
    """
    Desc:
        Use scikit-learn to calculate the Brier score and fixed-threshold
        classification metrics. The threshold is not optimized on the test set.
    Args:
        y, p (numpy.ndarray): Labels and raw predicted probabilities.
        n_bins (int): Fixed number of bins used for reliability and ECE.
        threshold (float): Prespecified reference threshold, with a default of
        0.5; the classification rule is score > threshold.
        reference_probability (float): Positive proportion in the original
        test set, used only as a descriptive constant baseline.
    """
    bins, ece = reliability(y, p, n_bins)
    predicted = p > threshold
    brier = brier_score_loss(y, p)
    reference = brier_score_loss(y, np.full(len(y), reference_probability))
    return np.array([brier, ece, reference, brier-reference,
                     recall_score(y, predicted, pos_label=1, zero_division=0),
                     recall_score(y, predicted, pos_label=0, zero_division=0),
                     precision_score(y, predicted, zero_division=0)]), bins


def ppv(sensitivity, specificity, prevalences):
    """
    Desc:
        Project PPV at assumed pathogenic candidate-variant proportions using
        Bayes' theorem. These projections are not treated as clinically
        observed values.
    Args:
        sensitivity, specificity (float or numpy.ndarray): Performance at the
        fixed threshold.
        prevalences (numpy.ndarray): Assumed positive proportions, not
        population disease prevalences.
    """
    numerator = np.asarray(sensitivity)[..., None]*prevalences
    denominator = numerator+(1-np.asarray(specificity)[..., None])*(1-prevalences)
    return np.divide(numerator, denominator, out=np.full_like(denominator,np.nan,dtype=float), where=denominator>0)


def interval(values):
    """
    Desc:
        Calculate a percentile interval from finite bootstrap values and record
        the number of valid replicates.
    Args:
        values (array-like): Bootstrap replicate values for one metric.
    """
    values = np.asarray(values); values = values[np.isfinite(values)]
    return (*np.quantile(values,[.025,.975]),len(values)) if len(values) else (np.nan,np.nan,0)


def evaluate(df, repetitions, seed, n_bins, threshold, prevalences):
    """
    Desc:
        Perform paired bootstrap resampling of all available TranSyn test
        variants, using gene_id as the cluster. Bin boundaries and the threshold
        remain fixed; no model is retrained and no calibrator is fitted.
    Args:
        df (pandas.DataFrame): Valid test set containing variant38, label,
        gene_id, and TranSyn columns.
        repetitions, seed (int): Number of valid bootstrap replicates and the
        random seed.
        n_bins (int): Fixed number of equal-width bins.
        threshold (float): Reference threshold, not a threshold selected on a
        validation set.
        prevalences (numpy.ndarray): Assumed pathogenic-proportion scenarios.
    """
    y, p = df.label.to_numpy(int), df.TranSyn.to_numpy(float)
    reference = float(y.mean())
    point, bins = measures(y,p,n_bins,threshold,reference)
    groups = list(df.reset_index(drop=True).groupby('gene_id',sort=False).indices.values())
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions,len(METRICS)))
    observed_draws = np.full((repetitions,n_bins),np.nan)
    attempts = 0
    for i in range(repetitions):
        while True:
            attempts += 1
            if attempts > repetitions*100: raise ValueError('Too many single-class bootstrap draws')
            idx = np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
            if np.unique(y[idx]).size == 2: break
        draws[i], sampled_bins = measures(y[idx],p[idx],n_bins,threshold,reference)
        observed_draws[i] = sampled_bins.observed_fraction
    summary = []
    for j, metric in enumerate(METRICS):
        lo,hi,n = interval(draws[:,j])
        summary.append(dict(metric=metric,estimate=point[j],ci_low=lo,ci_high=hi,valid_replicates=n,
                            n=len(df),positive=int(y.sum()),n_genes=len(groups),
                            bootstrap_repetitions=repetitions,seed=seed,n_bins=n_bins,
                            threshold=threshold,reference_probability=reference,
                            rejected_single_class_draws=attempts-repetitions))
    for j in range(n_bins):
        lo,hi,n = interval(observed_draws[:,j])
        bins.loc[j,['ci_low','ci_high','valid_replicates']] = [lo,hi,n]
    scenario_point = ppv(point[4],point[5],prevalences)
    scenario_draws = ppv(draws[:,4],draws[:,5],prevalences)
    scenarios = []
    for j, prior in enumerate(prevalences):
        lo,hi,n = interval(scenario_draws[:,j])
        scenarios.append(dict(prevalence=prior,ppv=scenario_point[j],ci_low=lo,ci_high=hi,
                              valid_replicates=n,threshold=threshold,
                              threshold_source='prespecified_reference_not_validation_selected'))
    tn,fp,fn,tp = confusion_matrix(y,p>threshold,labels=[0,1]).ravel()
    counts = dict(threshold=threshold,rule='score > threshold',tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                  threshold_source='prespecified_reference_not_validation_selected')
    return pd.DataFrame(summary),bins,pd.DataFrame(scenarios),counts

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--bins',type=int,default=5)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--prevalences',type=float,nargs='+',default=[.001,.01,.05,.1])
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args()
    if args.bootstrap<100 or args.bins<2 or not 0<args.threshold<1:parser.error('Invalid bootstrap/bins/threshold')
    priors=np.asarray(args.prevalences)
    if np.any((priors<=0)|(priors>=1)):parser.error('Prevalences must be between 0 and 1')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]; metadata=dict(
                            bootstrap_repetitions=args.bootstrap,
                            seed=args.seed,
                            n_bins=args.bins,
                            threshold=args.threshold,
                            threshold_rule='score > threshold',
                            threshold_status='prespecified_reference_not_validation_selected',
                            validation_available=False,
                            recalibration_fitted=False,unit='gene_id',
                            prevalences=priors.tolist(),
                            python_executable=sys.executable,
                            packages={p:version(p) for p in ['scikit-learn','scipy','numpy','pandas']},
                            datasets={}
                         )
    for name in ['Test.set.I','Test.set.II']:
        path=ROOT / f'{name}.benchmark.txt'
        df=pd.read_csv(path,sep='\t')[['variant38','label','transcript_id','gene_id','TranSyn']]
        if df.variant38.duplicated().any() or df.gene_id.isna().any():raise ValueError('Duplicate variants or missing genes')
        if df.label.isna().any() or set(df.label.unique())!={0,1}:raise ValueError('Invalid labels')
        valid=np.isfinite(df.TranSyn); excluded=int((~valid).sum());df=df.loc[valid].reset_index(drop=True)
        if not df.TranSyn.between(0,1).all():raise ValueError('Raw score outside [0,1]; do not silently normalize')
        print(name, len(df), 'variants;',df.gene_id.nunique(),'genes',flush=True)
        summary,bins,scenarios,counts=evaluate(df,args.bootstrap,args.seed,args.bins,args.threshold,priors)
        for suffix,table in [('summary',summary),('reliability',bins),('prevalence',scenarios),('memberships',df)]:
            table.to_csv(args.output/f'{name}.{suffix}.tsv',sep='\t',index=False)
        (args.output/f'{name}.threshold.json').write_text(json.dumps(counts,indent=2)+'\n')
        metadata['datasets'][name]=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      n=len(df),positive=int(df.label.sum()),genes=int(df.gene_id.nunique()),excluded_scores=excluded)
        rows.append((name,summary,bins,scenarios,counts))
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    main()
