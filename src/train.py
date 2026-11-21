
import os
import json
import csv
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataset import FundusDataset
from src.models import get_model
from src.transforms import get_train_transforms, get_eval_transforms
from src.utils import generate_training_graphs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint_safely(state, path):
    """Save checkpoint safely using a temporary file."""
    temp_path = path + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    training,
    accumulation_steps,
    use_amp
):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, labels, patient_ids) in enumerate(loader):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):

            with autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:

                group_start = (
                    step // accumulation_steps
                ) * accumulation_steps

                group_size = min(
                    accumulation_steps,
                    len(loader) - group_start
                )

                scaler.scale(
                    loss / group_size
                ).backward()

                if (
                    (step + 1) % accumulation_steps == 0
                    or (step + 1) == len(loader)
                ):
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)

        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    return average_loss, accuracy, auc


def main(args):

    with open(args.config, "r") as f:
        config = json.load(f)

    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and config["mixed_precision"]
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Mixed precision:", use_amp)

    eye_folder = f"{args.eye}_eye"

    # Separate original experiments from stacking experiments
    train_patient_ids = None

    if args.stacking:

        split_path = "configs/stacking_split.json"

        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Stacking split not found: {split_path}"
            )

        with open(split_path, "r") as f:
            stacking_split = json.load(f)

        train_patient_ids = stacking_split["base_train_ids"]

        output_root = "results_stacking"
        checkpoint_root = "checkpoints_stacking"

        print(
            "\nSTACKING MODE ENABLED"
            f"\nBase CNN training patients: {len(train_patient_ids)}"
        )

    else:
        output_root = "results"
        checkpoint_root = "checkpoints"

    output_dir = os.path.join(
        output_root,
        args.model,
        eye_folder
    )

    checkpoint_dir = os.path.join(
        checkpoint_root,
        args.model,
        eye_folder
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        "training_log.csv"
    )

    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        "best_model.pth"
    )

    last_checkpoint_path = os.path.join(
        checkpoint_dir,
        "last_checkpoint.pth"
    )

    # Datasets

    train_dataset = FundusDataset(
        "data/data_info.json",
        "data/images",
        "train",
        args.eye,
        config["image_size"],
        get_train_transforms(config["image_size"]),
        patient_ids=train_patient_ids
    )

    val_dataset = FundusDataset(
        "data/data_info.json",
        "data/images",
        "val",
        args.eye,
        config["image_size"],
        get_eval_transforms(config["image_size"])
    )

    print("\nTraining samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # Dedicated generator for reproducible shuffling

    train_generator = torch.Generator()
    train_generator.manual_seed(config["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda")
    )

    # Model

    model = get_model(
        args.model,
        pretrained=config["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_auc = -1.0
    start_epoch = 0

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
        "epoch_seconds"
    ]

    # Resume from latest completed epoch

    if args.resume:

        if not os.path.exists(last_checkpoint_path):
            raise FileNotFoundError(
                "No resumable checkpoint found: "
                + last_checkpoint_path
            )

        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["completed_epoch"]
        best_auc = checkpoint["best_val_auc"]

        # Restore random-number-generator states

        train_generator.set_state(
            checkpoint["train_generator_state"].cpu()
        )

        random.setstate(
            checkpoint["python_rng_state"]
        )

        np.random.set_state(
            checkpoint["numpy_rng_state"]
        )

        torch.set_rng_state(
            checkpoint["torch_rng_state"].cpu()
        )

        if device.type == "cuda":

            torch.cuda.set_rng_state_all(
                [
                    state.cpu()
                    for state in checkpoint["cuda_rng_states"]
                ]
            )

        print(
            f"\nResuming from completed epoch {start_epoch}"
            f"\nBest validation AUC so far: {best_auc:.4f}"
        )

        csv_mode = "a"

    else:

        if os.path.exists(last_checkpoint_path):
            raise FileExistsError(
                "A previous resumable checkpoint exists. "
                "Use --resume or archive it before starting fresh."
            )

        csv_mode = "w"

    # Training

    with open(csv_path, csv_mode, newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        if csv_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, config["epochs"]):

            start_time = time.time()

            # Training epoch

            train_loss, train_acc, train_auc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                True,
                config["gradient_accumulation_steps"],
                use_amp
            )

            # Validation epoch

            with torch.no_grad():

                val_loss, val_acc, val_auc = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler,
                    False,
                    config["gradient_accumulation_steps"],
                    use_amp
                )

            epoch_time = time.time() - start_time

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch [{epoch + 1}/{config['epochs']}]"
                f"\nTrain Loss: {train_loss:.4f}"
                f" | Train Acc: {train_acc:.4f}"
                f" | Train AUC: {train_auc:.4f}"
                f"\nVal Loss: {val_loss:.4f}"
                f" | Val Acc: {val_acc:.4f}"
                f" | Val AUC: {val_auc:.4f}"
                f"\nTime: {epoch_time:.1f}s"
            )

            # Write training log

            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "learning_rate": current_lr,
                "epoch_seconds": epoch_time
            })

            csvfile.flush()
            os.fsync(csvfile.fileno())

            # Save best model based on validation AUC

            if val_auc > best_auc:

                best_auc = val_auc

                save_checkpoint_safely({
                    "epoch": epoch + 1,
                    "model_name": args.model,
                    "eye": args.eye,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_auc,
                    "config": config,
                    "stacking_mode": args.stacking
                }, best_checkpoint_path)

                print("Best checkpoint saved!")

            # Save latest resumable checkpoint

            last_state = {
                "completed_epoch": epoch + 1,
                "model_name": args.model,
                "eye": args.eye,

                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),

                "best_val_auc": best_auc,

                "train_generator_state": train_generator.get_state(),

                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),

                "torch_rng_state": torch.get_rng_state(),

                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None
                ),

                "config": config,
                "stacking_mode": args.stacking
            }

            save_checkpoint_safely(
                last_state,
                last_checkpoint_path
            )

            print("Latest resumable checkpoint saved!")

    # Generate graphs after training

    generate_training_graphs(
        csv_path,
        output_dir
    )

    print("\nTraining completed.")
    print("Best validation AUC:", best_auc)
    print("CSV saved:", csv_path)
    print("Best checkpoint:", best_checkpoint_path)
    print("Recovery checkpoint:", last_checkpoint_path)


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
        "--config",
        type=str,
        default="configs/config.json"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest saved checkpoint"
    )

    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Train a separate base model for holdout stacking"
    )

    args = parser.parse_args()

    main(args)
