import os
import os.path as osp
import random

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASSES = (
    "background",
    "oil",
    "others",
    "water",
)
PALETTE = [
    [0, 0, 0],
    [255, 0, 124],
    [255, 204, 51],
    [51, 221, 255],
]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (512, 960)
TEST_IMG_SIZE = (512, 960)


def get_training_transform():
    return albu.Compose(
        [
            albu.Resize(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
            albu.Sharpen(),
            albu.Normalize(),
        ]
    )


def train_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def get_val_transform():
    return albu.Compose(
        [
            albu.Resize(height=INPUT_IMG_SIZE[0], width=INPUT_IMG_SIZE[1]),
            albu.Normalize(),
        ]
    )


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def test_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def uavrgb_collate_fn(batch):
    return {
        "img": torch.stack([b["img"] for b in batch], 0),
        "gt_semantic_seg": torch.stack([b["gt_semantic_seg"] for b in batch], 0),
        "img_id": [b["img_id"] for b in batch],
    }


def rgb_mask_to_index(mask_rgb: np.ndarray, palette=PALETTE, default_ignore=255):
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        return mask_rgb.astype(np.uint8)
    h, w, _ = mask_rgb.shape
    index_mask = np.full((h, w), default_ignore, dtype=np.uint8)
    for idx, color in enumerate(palette):
        match = (
            (mask_rgb[:, :, 0] == color[0])
            & (mask_rgb[:, :, 1] == color[1])
            & (mask_rgb[:, :, 2] == color[2])
        )
        index_mask[match] = idx
    return index_mask


class UAVRGBDataset(Dataset):
    def __init__(
        self,
        data_root="data/UAVRGB/Train",
        mode="train",
        img_dir="images",
        mask_dir="masks",
        img_suffix=".jpg",
        mask_suffix=".png",
        transform=train_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img, mask = self.load_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            img, mask = np.array(img), np.array(mask)
        img = torch.as_tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).contiguous()
        mask = torch.as_tensor(mask.copy(), dtype=torch.long).contiguous()
        img_id = self.img_ids[index]
        return {"img": img, "gt_semantic_seg": mask, "img_id": img_id}

    @staticmethod
    def get_img_ids(data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list), (
            f"images({len(img_filename_list)}) and masks({len(mask_filename_list)}) count mismatch"
        )
        img_ids = [str(osp.splitext(id)[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert("RGB")
        mask_rgb = np.array(Image.open(mask_name).convert("RGB"))
        mask = rgb_mask_to_index(mask_rgb)
        mask = Image.fromarray(mask)
        return img, mask
