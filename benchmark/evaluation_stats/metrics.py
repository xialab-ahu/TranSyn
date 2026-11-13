"""
@description: Calculate benchmark evaluation metrics.
@author: chen ye
@email: q23101020@stu.ahu.edu.cn
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, recall_score
from scipy.integrate import trapezoid
from evaluation_config import FRACTIONS

def metrics(y, scores):
    """
    Desc:
        Calculate the AUROC and AUPR evaluation metrics.
    Args:
        y, ground-truth labels.
        scores, predicted scores.
    """
    return np.array([roc_auc_score(y, scores), average_precision_score(y, scores)])


def hsr_auc(y, scores, sensitivity=.95):
    """
    Desc:
        Calculate the normalized area in the region where TPR is at least the target sensitivity: integrate max(TPR - s, 0) over FPR and divide by 1 - s. Use linear ROC interpolation and handle segments crossing the boundary exactly. This is not an FPR-restricted partial AUROC and does not include random-baseline correction.

    Args:
        y (numpy.ndarray): One-dimensional labels encoded as 0 or 1 and containing both classes.
        scores (numpy.ndarray): Finite scores with the same length as y; higher values indicate greater pathogenicity.
        sensitivity (float): Lower bound of the high-sensitivity region, strictly between 0 and 1; default is 0.95.
    """
    if not 0 < sensitivity < 1:
        raise ValueError('Sensitivity must lie strictly between 0 and 1')
    if np.unique(y).size != 2:
        raise ValueError('Both classes required')
    # Let scikit-learn handle ROC thresholds and tied scores; only add the target TPR boundary.
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    index = int(np.searchsorted(tpr, sensitivity, side='left'))
    if 0 < index < len(tpr) and tpr[index-1] < sensitivity < tpr[index]:
        boundary = np.interp(sensitivity, tpr[index-1:index+1], fpr[index-1:index+1])
        fpr = np.insert(fpr, index, boundary)
        tpr = np.insert(tpr, index, sensitivity)
    # Use SciPy for trapezoidal integration; max_fpr cannot replace the TPR lower bound and normalization required by hsr-AUC.
    return float(trapezoid(np.maximum(tpr-sensitivity, 0), x=fpr)/(1-sensitivity))


def recall_at_k(y_true, y_score, k):
    """
    Desc:
        Follow the ordering rule from the original calculate_metrics.py and calculate the proportion of all positives captured among the top k variants.

    Args:
        y_true (array-like): One-dimensional ground-truth labels encoded as 0 or 1.
        y_score (array-like): Prediction scores with the same length as the labels; higher scores rank earlier.
        k (int): Number of top-ranked variants to select. Existing slicing behavior is retained, and callers should provide a non-negative value.
    """
    # Keep the ordering and slicing rules consistent with the original calculate_metrics.py.
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    idx = np.argsort(y_score)[::-1]
    predicted = np.zeros(len(y_true), dtype=int)
    predicted[idx[:k]] = 1
    return float(recall_score(y_true, predicted, zero_division=0))


def recall_at_percent(y_true, y_score, percent=.01):
    """
    Desc:
        Convert the top fraction to k and call recall_at_k, rounding down while selecting at least one variant.
    Args:
        y_true (array-like): One-dimensional ground-truth labels encoded as 0 or 1.
        y_score (array-like): Prediction scores with the same length as the labels.
        percent (float): Top fraction rather than a percentage value; for example, 0.01 represents 1%. Default is 0.01.
    """
    return recall_at_k(y_true, y_score, max(1, int(len(y_true)*percent)))


def measure(y, score, dataset):
    """
    Desc:
        Calculate the metrics required for the test set on a fixed sample set. Test I returns AUROC and hsr-AUC; Test II returns AP and Recall from 1% through 20%.
    Args:
        y (numpy.ndarray): One-dimensional binary labels containing both classes.
        score (numpy.ndarray): Corresponding prediction scores; higher values indicate greater pathogenicity.
        dataset (str): I or II, corresponding to the two test sets.
    """
    primary = roc_auc_score(y, score) if dataset == 'I' else average_precision_score(y, score)
    if dataset == 'I':
        return np.array([primary, hsr_auc(y, score)])
    # Retain the original Top-k ordering and rounding behavior; organize the 20 thresholds into a multilabel matrix,
    # and use scikit-learn to calculate Recall for all columns at once, avoiding the overhead of 20 separate calls.
    ranks = np.empty(len(score), dtype=int)
    ranks[np.argsort(score)[::-1]] = np.arange(len(score))
    ks = np.array([max(1, int(len(y)*p)) for p in FRACTIONS])
    predicted = ranks[:, None] < ks[None, :]
    targets = np.broadcast_to(y[:, None], predicted.shape)
    recalls = recall_score(targets, predicted, average=None, zero_division=0)
    return np.r_[primary, recalls]

