import os
import os.path as osp

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

IGNORE_INDEX = 255
COLOR_TO_LABEL = {
    (0, 0, 0): 0,
    (255, 0, 124): 1,
    (255, 204, 51): 2,
    (51, 221, 255): 3,
}

ORIGIN_IMG_SIZE = (1080, 1920)
INPUT_IMG_SIZE = (512, 960)
TEST_IMG_SIZE = (512, 960)


def build_resize_transform(img_size):
    if img_size is None or tuple(img_size) == ORIGIN_IMG_SIZE:
        return []
    return [albu.Resize(height=img_size[0], width=img_size[1])]


def get_training_transform(img_size=INPUT_IMG_SIZE):
    train_transform = build_resize_transform(img_size) + [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
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
    val_transform = build_resize_transform(img_size) + [albu.Normalize()]
    return albu.Compose(val_transform)


def val_aug(img, mask, img_size=TEST_IMG_SIZE):
    img, mask = np.array(img), np.array(mask, dtype=np.uint8)
    aug = get_val_transform(img_size=img_size)(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def test_aug(img, mask, img_size=TEST_IMG_SIZE):
    return val_aug(img, mask, img_size=img_size)


def rgb_mask_to_label(mask_rgb):
    mask_rgb = np.asarray(mask_rgb, dtype=np.uint8)
    label_mask = np.full(mask_rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)

    for color, class_id in COLOR_TO_LABEL.items():
        color_mask = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[color_mask] = class_id

    unknown_mask = label_mask == IGNORE_INDEX
    if np.any(unknown_mask):
        unknown_colors = np.unique(mask_rgb[unknown_mask].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown UAVRGB mask colors found: {unknown_colors.tolist()}")

    return label_mask


class UAVRGBDataset(Dataset):
    SPLIT_MAP = {
        "train": "Train",
        "val": "Val",
        "test": "Test",
    }

    def __init__(
        self,
        data_root="data/UAVRGB",
        mode="train",
        img_dir="images",
        mask_dir="masks",
        img_suffix=".jpg",
        mask_suffix=".png",
        transform=test_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
        ignore_index=IGNORE_INDEX,
    ):
        if mode not in self.SPLIT_MAP:
            raise ValueError(f"Unsupported UAVRGB mode: {mode}")
        self.data_root = data_root
        self.mode = mode
        self.split_dir = self.resolve_split_dir()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.ignore_index = ignore_index
        self.img_ids = self.get_img_ids()

    def resolve_split_dir(self):
        expected_split = self.SPLIT_MAP[self.mode]
        root_name = osp.basename(osp.normpath(self.data_root))
        if root_name.lower() == expected_split.lower():
            return self.data_root
        return osp.join(self.data_root, expected_split)

    def __getitem__(self, index):
        img, mask = self.load_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        return dict(img_id=img_id, img=img, gt_semantic_seg=mask)

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self):
        img_root = osp.join(self.split_dir, self.img_dir)
        mask_root = osp.join(self.split_dir, self.mask_dir)
        img_ids = sorted(osp.splitext(name)[0] for name in os.listdir(img_root) if name.endswith(self.img_suffix))
        mask_ids = sorted(osp.splitext(name)[0] for name in os.listdir(mask_root) if name.endswith(self.mask_suffix))
        if img_ids != mask_ids:
            raise ValueError(f"UAVRGB image ids and mask ids do not match under split {self.split_dir}.")
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.split_dir, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.split_dir, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert("RGB")
        mask_rgb = Image.open(mask_name).convert("RGB")
        mask = rgb_mask_to_label(mask_rgb)
        return img, mask
