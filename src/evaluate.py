
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)


def evaluate_predictions(args):

    # Load predictions
    df = pd.read_csv(args.predictions)

    y_true = df["label"].values
    y_prob = df["probability"].values

    # Convert probabilities into binary predictions
    y_pred = (y_prob >= args.threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    # Print results
    print("\n========== EVALUATION RESULTS ==========")

    print(f"Patients:  {len(df)}")
    print(f"Threshold: {args.threshold}")

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Thickened"],
            zero_division=0
        )
    )

    # Create output directory
    output_dir = os.path.dirname(args.predictions)
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Thickened"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(
        output_dir, "confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(
        output_dir, "roc_curve.png"
    )

    plt.savefig(roc_path, dpi=300)
    plt.close()

    print("\nSaved evaluation graphs:")
    print(cm_path)
    print(roc_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=str,
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    evaluate_predictions(args)
