import json
import os
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from PIL import Image


class HouseholdWasteDataset:
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        json_split: str = "household_waste_split.json",
        partial: float = 1,
    ):
        self.root_dir = root_dir
        self.transform = transform if transform else get_transforms()
        self.label_mapping = {}  # Add a mapping dictionary
        self.inverse_label_mapping = {}  # Add an inverse mapping dictionary

        if partial < 1:
            json_split = f"household_waste_split_{partial}.json"

        # if json file is not available, create the split and store it to json file
        if not os.path.exists(json_split):
            # np.random.seed(2024)
            print(f"JSON file not found. Creating split and store it to {json_split}")
            # get img category
            img_label = os.listdir(root_dir)

            # only use partial of the dataset category
            partial_val = int(partial * len(img_label))
            img_label = img_label[:partial_val]

            # if .DS_Store is available, remove it
            if ".DS_Store" in img_label:
                img_label.remove(".DS_Store")
            print(f"Length of img_label: {len(img_label)}")
            print(f"img_label: {img_label}")

            img_ext = ["jpg", "jpeg", "png"]
            self.train_img_paths = []
            self.val_img_paths = []

            for each_label in img_label:
                all_img_paths = []

                for ext in img_ext:
                    all_img_paths.extend(
                        glob(
                            os.path.join(root_dir, each_label, f"**/*.{ext}"),
                            recursive=True,
                        )
                    )

                # shuffle and split into train and test set
                np.random.shuffle(all_img_paths)
                split_idx = int(0.8 * len(all_img_paths))
                self.train_img_paths.extend(all_img_paths[:split_idx])
                self.val_img_paths.extend(all_img_paths[split_idx:])

            json_object = {
                "train": self.train_img_paths,
                "val": self.val_img_paths,
            }

            # store this split to external single json file
            with open(json_split, "w") as f:
                json.dump(json_object, f)

        print(f"JSON file found. Loading split from {json_split}")
        with open(json_split, "r") as f:
            json_object = json.load(f)

        self.train_img_paths = json_object["train"]
        self.val_img_paths = json_object["val"]

        if split == "train":
            self.img_paths = self.train_img_paths
        elif split == "val":
            self.img_paths = self.val_img_paths

        # Initialize label mapping
        self.init_label_mapping()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 1. load image using PIL
        img = Image.open(img_path)
        img = np.array(img)
        label = self.extract_label(img_path)
        # 2. apply any transformation (optional)
        if self.transform:
            img = self.transform(image=img)["image"]

        # 3. Convert the image to tensor
        img = mx.array(img)
        return img, label

    def extract_label(self, img_path):
        # Split the path and get the desired directory as label
        parts = img_path.split(os.sep)
        label = parts[-3]
        return self.label_mapping[label]

    def init_label_mapping(self):
        unique_labels = set()
        for path in self.img_paths:
            label = path.split(os.sep)[-3]
            unique_labels.add(label)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_mapping = {
            idx: label for label, idx in self.label_mapping.items()
        }


def get_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=True,
            ),
        ]
    )


if __name__ == "__main__":
    dataset_hhwd = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="train",
        transform=get_transforms(),
        partial=0.2,
    )
    dataset_hhwd_test = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="val",
        transform=get_transforms(),
        partial=0.2,
    )

    intersection = set(dataset_hhwd.img_paths).intersection(
        set(dataset_hhwd_test.img_paths)
    )
    print(f"Intersection between train and test dataset: {len(intersection)}")
    print(f"Intersection: {intersection}")
    print(
        f"Train dataset: {len(dataset_hhwd)} | Test dataset: {len(dataset_hhwd_test)}"
    )

    # Preview the image
    rnd_idx = np.random.randint(len(dataset_hhwd))
    sample_img, sample_label = dataset_hhwd[rnd_idx]
    # sample_img = sample_img.transpose(1, 2, 0)
    # renormalize the image
    sample_img = (sample_img * mx.array([0.229, 0.224, 0.225])) + mx.array(
        [0.485, 0.456, 0.406]
    )
    plt.figure(figsize=(5, 5))
    plt.imshow(sample_img)
    plt.title(f"Label: {dataset_hhwd.inverse_label_mapping[sample_label]}")
    plt.axis("off")
    plt.show()
