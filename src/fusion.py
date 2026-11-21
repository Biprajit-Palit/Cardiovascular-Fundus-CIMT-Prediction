
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

MODELS = ["resnet18", "resnext50", "vit_tiny"]

BASE_DIR = "predictions"
OUTPUT_DIR = "fusion_results"


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
        raise ValueError(f"Patient mismatch in {split} predictions")

    if not (merged["label_right"] == merged["label_left"]).all():
        raise ValueError(f"Label mismatch between eyes in {split}")

    merged = merged.rename(columns={
        "label_right": "label"
    })

    return merged[[
        "patient_id",
        "label",
        "probability_right",
        "probability_left"
    ]]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in MODELS:
        print(f"\n{'=' * 50}")
        print(f"FUSION MODEL: {model_name}")
        print("=" * 50)

        # Load validation predictions for training the meta-model
        val_right = load_predictions(model_name, "right", "val")
        val_left = load_predictions(model_name, "left", "val")

        val_data = combine_eyes(val_right, val_left, "validation")

        # Load test predictions for final evaluation
        test_right = load_predictions(model_name, "right", "test")
        test_left = load_predictions(model_name, "left", "test")

        test_data = combine_eyes(test_right, test_left, "test")

        X_train = val_data[
            ["probability_right", "probability_left"]
        ]
        y_train = val_data["label"]

        X_test = test_data[
            ["probability_right", "probability_left"]
        ]

        # Fit Logistic Regression using validation predictions only
        fusion_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        fusion_model.fit(X_train, y_train)

        # Predict fused probabilities on the untouched test set
        test_probabilities = fusion_model.predict_proba(X_test)[:, 1]

        output = pd.DataFrame({
            "patient_id": test_data["patient_id"],
            "label": test_data["label"],
            "probability": test_probabilities
        })

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        prediction_path = os.path.join(
            model_dir, "test_fusion_predictions.csv"
        )

        output.to_csv(prediction_path, index=False)

        # Save fitted model for reproducibility
        joblib.dump(
            fusion_model,
            os.path.join(model_dir, "logistic_regression.joblib")
        )

        # Save model coefficients
        coefficients = pd.DataFrame({
            "feature": ["right_eye_probability", "left_eye_probability"],
            "coefficient": fusion_model.coef_[0]
        })

        coefficients.to_csv(
            os.path.join(model_dir, "coefficients.csv"),
            index=False
        )

        print(f"Validation patients used for fitting: {len(val_data)}")
        print(f"Test patients predicted: {len(test_data)}")
        print(f"Intercept: {fusion_model.intercept_[0]:.6f}")
        print(f"Right-eye coefficient: {fusion_model.coef_[0][0]:.6f}")
        print(f"Left-eye coefficient: {fusion_model.coef_[0][1]:.6f}")
        print(f"Saved: {prediction_path}")

    print("\nAll three Logistic Regression fusion models completed!")


if __name__ == "__main__":
    main()
