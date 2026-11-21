
import json
from sklearn.model_selection import train_test_split

with open("data/data_info.json", "r") as f:
    metadata = json.load(f)

patient_ids = []
labels = []

for patient_id, info in metadata.items():
    if info["group"] == 1:
        patient_ids.append(str(patient_id))
        labels.append(int(info["label"]))

base_ids, fusion_ids = train_test_split(
    patient_ids,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Store the split
split_data = {
    "base_train_ids": base_ids,
    "fusion_train_ids": fusion_ids
}

with open("configs/stacking_split.json", "w") as f:
    json.dump(split_data, f, indent=4)

# Verify label distributions
label_map = {
    str(pid): int(info["label"])
    for pid, info in metadata.items()
    if info["group"] == 1
}

for name, ids in [
    ("Base CNN training", base_ids),
    ("Fusion training", fusion_ids)
]:
    counts = {
        label: sum(label_map[pid] == label for pid in ids)
        for label in [0, 1]
    }
    print(f"{name}: {len(ids)} patients | Normal: {counts[0]} | Thickened: {counts[1]}")

print("\nSplit saved to configs/stacking_split.json")
