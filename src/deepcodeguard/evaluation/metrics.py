"""Evaluation helpers shared by every DeepCodeGuard experiment.

Protocol: pick the decision threshold on validation, then apply it unchanged to test.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def best_threshold(y_true, probs):
    """Threshold in [0.05, 0.95] that maximises F1 on the given (validation) data."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, probs >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, probs, threshold=0.5):
    """All metrics we report, for one set of predicted probabilities."""
    pred = (np.asarray(probs) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(set(probs)) > 1 else 0.5,
        "pr_auc": average_precision_score(y_true, probs),
    }


def val_then_test(y_val, val_probs, y_test, test_probs):
    """Tune the threshold on validation, then score validation and test with it."""
    t = best_threshold(y_val, val_probs)
    return t, evaluate(y_val, val_probs, t), evaluate(y_test, test_probs, t)
