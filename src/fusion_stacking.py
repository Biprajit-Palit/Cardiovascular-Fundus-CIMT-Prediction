
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

MODELS = ["resnet18", "resnext50", "vit_tiny"]

BASE_DIR = "predictions_stacking"
OUTPUT_DIR = "fusion_results_stacking"

FEATURES = ["probability_right", "probability_left"]


def load_predictions(model, eye, split):
    path = os.path.join(
        BASE_DIR, model, f"{eye}_eye", f"{split}_predictions.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)

    required = {"patient_id", "label", "probability"}

    if not required.issubset(df.columns):
        raise ValueError(f"Invalid CSV columns in {path}")

    if df["patient_id"].duplicated().any():
        raise ValueError(f"Duplicate patient IDs in {path}")

    return df[["patient_id", "label", "probability"]]


def combine_eyes(right, left, split):
    merged = right.merge(
        left,
        on="patient_id",
        how="inner",
        suffixes=("_right", "_left"),
        validate="one_to_one"
    )

    if len(merged) != len(right) or len(merged) != len(left):
        raise ValueError(f"Patient mismatch in {split}")

    if not (merged["label_right"] == merged["label_left"]).all():
        raise ValueError(f"Label mismatch between eyes in {split}")

    merged = merged.rename(columns={"label_right": "label"})

    return merged[
        [
            "patient_id",
            "label",
            "probability_right",
            "probability_left"
        ]
    ]


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def save_roc_curve(y_true, probabilities, model_name, split, output_dir):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_value = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr, tpr,
        label=f"{model_name} (AUC = {auc_value:.4f})"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - {split.capitalize()} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{split}_roc_curve.png"),
        dpi=300
    )
    plt.close()


def save_confusion_matrix(metrics, model_name, split, output_dir):
    cm = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} - {split.capitalize()} Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], ["Normal", "Thickened"])
    plt.yticks([0, 1], ["Normal", "Thickened"])

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i][j]),
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{split}_confusion_matrix.png"),
        dpi=300
    )
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for model_name in MODELS:
        print("\n" + "=" * 60)
        print(f"HOLDOUT STACKING FUSION: {model_name}")
        print("=" * 60)

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Load all three patient-level splits
        train_right = load_predictions(
            model_name, "right", "fusion_train"
        )
        train_left = load_predictions(
            model_name, "left", "fusion_train"
        )

        val_right = load_predictions(model_name, "right", "val")
        val_left = load_predictions(model_name, "left", "val")

        test_right = load_predictions(model_name, "right", "test")
        test_left = load_predictions(model_name, "left", "test")

        train_data = combine_eyes(
            train_right, train_left, "fusion_train"
        )
        val_data = combine_eyes(val_right, val_left, "validation")
        test_data = combine_eyes(test_right, test_left, "test")

        print(f"Fusion training patients: {len(train_data)}")
        print(f"Validation patients: {len(val_data)}")
        print(f"Test patients: {len(test_data)}")

        # Train Logistic Regression only on fusion_train
        X_train = train_data[FEATURES]
        y_train = train_data["label"]

        fusion_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        fusion_model.fit(X_train, y_train)

        # Save the fitted fusion model
        joblib.dump(
            fusion_model,
            os.path.join(model_dir, "logistic_regression.joblib")
        )

        # Save coefficients
        coefficients = pd.DataFrame({
            "feature": FEATURES,
            "coefficient": fusion_model.coef_[0]
        })

        coefficients.loc[len(coefficients)] = [
            "intercept",
            fusion_model.intercept_[0]
        ]

        coefficients.to_csv(
            os.path.join(model_dir, "coefficients.csv"),
            index=False
        )

        # Evaluate validation and test independently
        for split, data in [
            ("val", val_data),
            ("test", test_data)
        ]:
            X = data[FEATURES]
            y = data["label"]

            probabilities = fusion_model.predict_proba(X)[:, 1]

            metrics = calculate_metrics(y, probabilities)

            predictions = (probabilities >= 0.5).astype(int)

            output = pd.DataFrame({
                "patient_id": data["patient_id"],
                "label": y,
                "probability": probabilities,
                "predicted_label": predictions
            })

            output.to_csv(
                os.path.join(
                    model_dir, f"{split}_fusion_predictions.csv"
                ),
                index=False
            )

            save_roc_curve(
                y, probabilities, model_name, split, model_dir
            )

            save_confusion_matrix(
                metrics, model_name, split, model_dir
            )

            result = {
                "model": model_name,
                "split": split,
                "patients": len(data),
                **metrics
            }

            all_results.append(result)

            print(f"\n{split.upper()} RESULTS")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}" if isinstance(
                    value, float
                ) else f"{key}: {value}")

        print(f"\nSaved model outputs in: {model_dir}")

    # Save combined comparison table
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "all_fusion_metrics.csv"),
        index=False
    )

    print("\n" + "=" * 60)
    print("ALL THREE HOLDOUT STACKING MODELS COMPLETED")
    print("=" * 60)

    print(results_df.to_string(index=False))
    print(f"\nFinal results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
