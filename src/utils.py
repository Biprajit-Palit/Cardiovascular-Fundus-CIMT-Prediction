
import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_training_graphs(csv_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Read epoch-wise training history
    df = pd.read_csv(csv_path)

    epochs = df["epoch"]

    # 1. Accuracy curves
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        df["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        df["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "accuracy_curve.png"),
        dpi=300
    )

    plt.close()

    # 2. Loss curves
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        df["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        df["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "loss_curve.png"),
        dpi=300
    )

    plt.close()

    # 3. Validation ROC-AUC
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        df["val_auc"],
        label="Validation ROC-AUC"
    )

    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Validation ROC-AUC Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "roc_auc_curve.png"),
        dpi=300
    )

    plt.close()

    print("Training graphs generated successfully!")
    print("Saved in:", output_dir)
