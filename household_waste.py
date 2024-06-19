import json
import os
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import mlx
import mlx.core as mx
import numpy as np
from PIL import Image

np.random.seed(2024)

class HouseholdWasteDataset:
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        json_split: str = "household_waste_split.json",
    ):
        self.root_dir = root_dir
        self.transform = transform if transform else get_transforms()
        
        # if json file is not available, create the split and store it to json file
        if not os.path.exists(json_split):
            print(f"JSON file not found. Creating split and store it to {json_split}")
            # get img labels
            img_label = os.listdir(root_dir)
            
            img_ext = ["jpg", "jpeg", "png"]
            self.train_img_paths = []
            self.val_img_paths = []
            
            for each_label in img_label:
                all_img_paths = []
                
                for ext in img_ext:
                    all_img_paths.extend(glob(os.path.join(root_dir, each_label, f"**/*.{ext}"), recursive=True))
                    
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
            with open("household_waste_split.json", "w") as f:
                json.dump(json_object, f)
        
        # if json file is available, load the split from json file
        else:
            print(f"JSON file found. Loading split from {json_split}")
            with open(json_split, "r") as f:
                json_object = json.load(f)
                
            self.train_img_paths = json_object["train"]
            self.val_img_paths = json_object["val"]
        
        if split == "train":
            self.img_paths = self.train_img_paths
        elif split == "val":
            self.img_paths = self.val_img_paths
        
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
        # 4. Reshape from (H, W, C) to (C, H, W)
        img = img.transpose(2, 0, 1)
        return img, label
    
    def extract_label(self, img_path):
        # Split the path and get the desired directory as label
        parts = img_path.split(os.sep)
        return parts[-3]
    
def get_transforms():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            always_apply=True,
            ),
    ])
    
if __name__ == "__main__":
    dataset_hhwd = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="train",
        )
    dataset_hhwd_test = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="val",
        )
    
    intersection = set(dataset_hhwd.img_paths).intersection(set(dataset_hhwd_test.img_paths))
    print(f"Intersection between train and test dataset: {len(intersection)}")
    print(f"Intersection: {intersection}")
    
    # # check if there's any train dataset in the test dataset
    # full_path_train = [os.path.basename(path) for path in dataset_hhwd.img_paths]
    # full_path_test = [os.path.basename(path) for path in dataset_hhwd_test.img_paths]
    # print(f"Length of train dataset: {len(full_path_train)}")
    # print(f"Length of test dataset: {len(full_path_test)}")
    
    # # find intersection between train and test dataset
    # intersection = set(full_path_train).intersection(set(full_path_test))
    # print(f"Intersection between train and test dataset: {len(intersection)}")
    # print(f"Intersection: {intersection}")
    
    # Preview the image
    # sample_img, sample_label = dataset_hhwd[0]
    # plt.figure(figsize=(5, 5))
    # plt.imshow(sample_img.transpose(1, 2, 0))
    # plt.title(sample_label)
    # plt.axis("off")
    # plt.show()