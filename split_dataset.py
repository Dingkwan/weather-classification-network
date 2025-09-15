import os
import shutil
import random
from pathlib import Path

dataset_dir = "/Users/dingkwanmok/Desktop/test/dataset"   # Original dataset path
output_dir = "/Users/dingkwanmok/Desktop/test/split_dataset"  # Output Path
val_ratio = 0.2   # Validation set ratio
seed = 42

# Categories that need to be processed
valid_classes = {"cloudy", "foggy", "rainy", "shine", "sunrise"}

def split_dataset(dataset_dir, output_dir, val_ratio, seed):
    random.seed(seed)

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Create destination folder
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    for d in [train_dir, val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Iterate over each category
    for class_name in os.listdir(dataset_dir):
        if class_name.lower() not in valid_classes:  # Skip alien_test or other directories
            print(f"Skipping {class_name}")
            continue

        class_path = dataset_dir / class_name
        if not class_path.is_dir():
            continue

        images = list(class_path.glob("*"))
        random.shuffle(images)

        val_size = int(len(images) * val_ratio)
        val_images = images[:val_size]
        train_images = images[val_size:]

        # Copy to destination folder
        for split, split_images in [("train", train_images), ("val", val_images)]:
            split_dir = output_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy(img, split_dir / img.name)

        print(f"[{class_name}] Train: {len(train_images)}, Val: {len(val_images)}")

if __name__ == "__main__":
    split_dataset(dataset_dir=dataset_dir, output_dir=output_dir, val_ratio=val_ratio, seed=seed)