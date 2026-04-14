import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image


CLASSES = ("Sea Surface", "Oil Spill", "Look-alike", "Ship", "Land")
PALETTE = [
    [0, 0, 0],
    [0, 255, 255],
    [255, 0, 0],
    [153, 76, 0],
    [0, 153, 0],
]

ORIGIN_IMG_SIZE = (650, 1250)
INPUT_IMG_SIZE = (650, 1250)
TEST_IMG_SIZE = (650, 1250)


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


class M4DDataset(Dataset):
    def __init__(
        self,
        data_root="data/M4D",
        mode="train",
        img_dir="images",
        mask_dir="labels_1D",
        img_suffix=".jpg",
        mask_suffix=".png",
        transform=test_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
    ):
        self.data_root = data_root
        self.mode = mode
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids()

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
        img_root = osp.join(self.data_root, self.mode, self.img_dir)
        mask_root = osp.join(self.data_root, self.mode, self.mask_dir)
        img_ids = sorted(osp.splitext(name)[0] for name in os.listdir(img_root) if name.endswith(self.img_suffix))
        mask_ids = sorted(osp.splitext(name)[0] for name in os.listdir(mask_root) if name.endswith(self.mask_suffix))
        if img_ids != mask_ids:
            raise ValueError(f"M4D image ids and mask ids do not match under {self.mode}.")
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.mode, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mode, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name)
        return img, mask
