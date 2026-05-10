"""
evaluate.py
-----------
Evaluation utilities: compute classification metrics and save
results to CSV/JSON for reporting.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, average: str = "weighted") -> dict:
    """
    Returns Accuracy, Precision, Recall, F1 for a prediction.

    Parameters
    ----------
    average : 'weighted', 'macro', or 'binary'
    """
    return {
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
        "F1-Score" : round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
    }


def full_report(y_true, y_pred, label_encoder=None) -> str:
    """Return sklearn's detailed per-class classification report."""
    target_names = label_encoder.classes_ if label_encoder else None
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


# ---------------------------------------------------------------------------
# Aggregation & saving
# ---------------------------------------------------------------------------

def compare_classifiers(results: list[dict], dataset_name: str, out_dir: str) -> pd.DataFrame:
    """
    Compile a comparison table from a list of result dicts.

    Each dict must have keys: 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score'

    Saves a CSV to out_dir and returns the DataFrame.
    """
    df = pd.DataFrame(results)
    df = df.set_index("Classifier")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset_name}_comparison.csv")
    df.to_csv(csv_path)
    print(f"\n[✓] Comparison table saved → {csv_path}")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame, dataset_name: str, out_dir: str):
    """Bar chart comparing all classifiers across all metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = df[metrics].reset_index().melt(id_vars="Classifier",
                                              var_name="Metric",
                                              value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Metric", y="Score", hue="Classifier", ax=ax, palette="Set2")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Classifier Comparison — {dataset_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(out_dir, f"{dataset_name}_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[✓] Comparison plot saved → {path}")


def plot_confusion_matrix(y_true, y_pred, classifier_name: str,
                          dataset_name: str, label_encoder, out_dir: str):
    """Heatmap of confusion matrix for a single classifier."""
    labels = label_encoder.classes_ if label_encoder else None
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(cm)), max(5, len(cm))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {classifier_name} on {dataset_name}", fontsize=12)
    plt.tight_layout()

    safe_name = classifier_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    path = os.path.join(out_dir, f"{dataset_name}_{safe_name}_cm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[✓] Confusion matrix saved → {path}")
