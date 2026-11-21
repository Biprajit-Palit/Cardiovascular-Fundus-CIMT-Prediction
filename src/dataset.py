
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FundusDataset(Dataset):

    def __init__(
        self,
        metadata_path,
        image_dir,
        split,
        eye,
        image_size=256,
        transform=None,
        patient_ids=None
    ):

        self.image_dir = image_dir
        self.eye = eye

        split_mapping = {
            "train": 1,
            "val": 2,
            "test": 3
        }

        if split not in split_mapping:
            raise ValueError("Invalid split name")

        if eye not in ["right", "left"]:
            raise ValueError("Eye must be 'right' or 'left'")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        allowed_ids = (
            set(str(pid) for pid in patient_ids)
            if patient_ids is not None
            else None
        )

        self.samples = []

        for patient_id, info in metadata.items():

            if info["group"] != split_mapping[split]:
                continue

            if (
                allowed_ids is not None
                and str(patient_id) not in allowed_ids
            ):
                continue

            image_name = (
                info["right_eye"]
                if eye == "right"
                else info["left_eye"]
            )

            image_path = os.path.join(
                self.image_dir,
                image_name
            )

            self.samples.append({
                "patient_id": patient_id,
                "image_path": image_path,
                "label": int(info["label"])
            })

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return (
            image,
            label,
            sample["patient_id"]
        )
