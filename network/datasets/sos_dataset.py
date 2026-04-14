import os
import os.path as osp
import random

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASSES = ("Non-Oil Spill", "Oil Spill")
PALETTE = [
    [0, 0, 0],
    [255, 255, 255],
]

ORIGIN_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (256, 256)
TEST_IMG_SIZE = (256, 256)
SENSORS = ("palsar", "sentinel")


def get_training_transform(img_size=INPUT_IMG_SIZE):
    train_transform = [
        albu.Resize(height=img_size[0], width=img_size[1]),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.25,
        ),
        albu.Normalize(),
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask, img_size=INPUT_IMG_SIZE):
    img, mask = np.array(img), np.array(mask, dtype=np.uint8)
    aug = get_training_transform(img_size=img_size)(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def get_val_transform(img_size=TEST_IMG_SIZE):
    val_transform = [
        albu.Resize(height=img_size[0], width=img_size[1]),
        albu.Normalize(),
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask, img_size=TEST_IMG_SIZE):
    img, mask = np.array(img), np.array(mask, dtype=np.uint8)
    aug = get_val_transform(img_size=img_size)(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def test_aug(img, mask, img_size=TEST_IMG_SIZE):
    return val_aug(img, mask, img_size=img_size)


class SOSDataset(Dataset):
    def __init__(
        self,
        data_root="data/SOS",
        mode="train",
        split="train",
        sensors=SENSORS,
        val_ratio=0.1,
        split_seed=42,
        img_suffix=".jpg",
        mask_suffix=".png",
        transform=test_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
    ):
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported SOS mode: {mode}")
        if mode == "train" and split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported SOS train split: {split}")
        if mode == "test" and split not in {"test", "all"}:
            raise ValueError(f"Unsupported SOS test split: {split}")

        self.data_root = data_root
        self.mode = mode
        self.split = split
        self.sensors = tuple(sensors)
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.samples = self.get_samples()

    def __getitem__(self, index):
        img, mask = self.load_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.samples[index]["img_id"]
        return dict(img_id=img_id, img=img, gt_semantic_seg=mask)

    def __len__(self):
        return len(self.samples)

    def get_samples(self):
        samples = []
        for sensor in self.sensors:
            sensor_samples = self.get_sensor_samples(sensor)
            samples.extend(sensor_samples)
        if not samples:
            raise ValueError(f"No SOS samples found under {self.data_root} for {self.mode}/{self.split}.")
        return samples

    def get_sensor_samples(self, sensor):
        if self.mode == "train":
            sensor_root = osp.join(self.data_root, "train", sensor)
            img_paths = {
                name[:-8]: osp.join(sensor_root, name)
                for name in os.listdir(sensor_root)
                if name.endswith("_sat" + self.img_suffix)
            }
            mask_paths = {
                name[:-9]: osp.join(sensor_root, name)
                for name in os.listdir(sensor_root)
                if name.endswith("_mask" + self.mask_suffix)
            }
        else:
            img_root = osp.join(self.data_root, "test", sensor, "sat")
            mask_root = osp.join(self.data_root, "test", sensor, "gt")
            img_paths = {
                name[:-8]: osp.join(img_root, name)
                for name in os.listdir(img_root)
                if name.endswith("_sat" + self.img_suffix)
            }
            mask_paths = {
                name[:-9]: osp.join(mask_root, name)
                for name in os.listdir(mask_root)
                if name.endswith("_mask" + self.mask_suffix)
            }

        img_ids = sorted(img_paths)
        mask_ids = sorted(mask_paths)
        if img_ids != mask_ids:
            raise ValueError(f"SOS image ids and mask ids do not match for sensor {sensor} in mode {self.mode}.")

        selected_ids = self.select_split_ids(img_ids)
        return [
            {
                "sensor": sensor,
                "sample_id": sample_id,
                "img_id": f"{sensor}_{sample_id}",
                "img_path": img_paths[sample_id],
                "mask_path": mask_paths[sample_id],
            }
            for sample_id in selected_ids
        ]

    def select_split_ids(self, sample_ids):
        if self.mode == "test" or self.split in {"all", "test"}:
            return sample_ids

        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"SOS val_ratio must be in (0, 1), but got {self.val_ratio}.")

        shuffled_ids = list(sample_ids)
        random.Random(self.split_seed).shuffle(shuffled_ids)
        val_count = max(1, int(round(len(shuffled_ids) * self.val_ratio)))
        val_ids = set(shuffled_ids[:val_count])

        if self.split == "val":
            return sorted(val_ids)
        return [sample_id for sample_id in sample_ids if sample_id not in val_ids]

    def load_img_and_mask(self, index):
        sample = self.samples[index]
        img = Image.open(sample["img_path"]).convert("RGB")
        mask = np.array(Image.open(sample["mask_path"]).convert("L"), dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8)
        return img, mask
