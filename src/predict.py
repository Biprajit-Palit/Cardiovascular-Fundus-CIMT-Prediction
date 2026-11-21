
import os
import csv
import json
import argparse

import torch
from torch.utils.data import DataLoader

from src.dataset import FundusDataset
from src.models import get_model
from src.transforms import get_eval_transforms


def generate_predictions(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    with open(args.config, "r") as f:
        config = json.load(f)

    eye_folder = f"{args.eye}_eye"

    # Select separate paths for stacking experiments
    if args.stacking:
        checkpoint_root = "checkpoints_stacking"
        prediction_root = "predictions_stacking"
    else:
        checkpoint_root = "checkpoints"
        prediction_root = "predictions"

    checkpoint_path = os.path.join(
        checkpoint_root,
        args.model,
        eye_folder,
        "best_model.pth"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load trained model
    model = get_model(
        args.model,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Prevent accidentally mixing original and stacking checkpoints
    if args.stacking and not checkpoint.get("stacking_mode", False):
        raise ValueError(
            "This checkpoint was not trained in stacking mode."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Best validation AUC:", checkpoint["best_val_auc"])

    # Select patients for prediction
    patient_ids = None

    if args.stacking and args.split == "fusion_train":

        with open("configs/stacking_split.json", "r") as f:
            stacking_split = json.load(f)

        patient_ids = stacking_split["fusion_train_ids"]

        dataset_split = "train"

    else:
        dataset_split = args.split

    dataset = FundusDataset(
        "data/data_info.json",
        "data/images",
        dataset_split,
        args.eye,
        config["image_size"],
        get_eval_transforms(config["image_size"]),
        patient_ids=patient_ids
    )

    if len(dataset) == 0:
        raise ValueError(
            "No patients found for the requested prediction split."
        )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda")
    )

    print("Patients to predict:", len(dataset))

    # Generate predictions
    predictions = []

    with torch.no_grad():

        for images, labels, patient_ids in loader:

            images = images.to(device, non_blocking=True)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            probabilities = probabilities.cpu().numpy()

            for patient_id, label, probability in zip(
                patient_ids,
                labels.numpy(),
                probabilities
            ):

                predictions.append({
                    "patient_id": str(patient_id),
                    "label": int(label),
                    "probability": float(probability)
                })

    # Output directory
    output_dir = os.path.join(
        prediction_root,
        args.model,
        eye_folder
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{args.split}_predictions.csv"
    )

    # Save predictions
    with open(output_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "label",
                "probability"
            ]
        )

        writer.writeheader()
        writer.writerows(predictions)

    print("\nPrediction generation completed!")
    print("Model:", args.model)
    print("Eye:", args.eye)
    print("Split:", args.split)
    print("Stacking mode:", args.stacking)
    print("Total patients:", len(predictions))
    print("Saved:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "resnext50", "vit_tiny"],
        required=True
    )

    parser.add_argument(
        "--eye",
        type=str,
        choices=["right", "left"],
        required=True
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "fusion_train", "val", "test"],
        default="val"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json"
    )

    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Use separate stacking checkpoints and prediction directories"
    )

    args = parser.parse_args()

    if args.stacking and args.split == "train":
        parser.error(
            "In stacking mode, use --split fusion_train "
            "instead of --split train."
        )

    generate_predictions(args)
