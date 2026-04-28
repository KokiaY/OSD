import os.path as osp

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASSES = tuple(f"Class_{index}" for index in range(1, 16))
PALETTE = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [192, 192, 192],
    [255, 165, 0],
]

IGNORE_INDEX = 255
ORIGIN_IMG_SIZE = (240, 240)
INPUT_IMG_SIZE = (256, 256)
TEST_IMG_SIZE = (256, 256)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RHORC_BANDS = ("492", "560", "665", "833")
MADOS_MULTISPECTRAL_BAND_SPECS = (
    ("442", "60"),
    ("492", "10"),
    ("560", "10"),
    ("665", "10"),
    ("704", "20"),
    ("740", "20"),
    ("783", "20"),
    ("833", "10"),
    ("865", "20"),
    ("1614", "20"),
    ("2202", "20"),
)
RHORC_BAND_ALIASES = {
    "442": ("442", "443"),
    "443": ("443", "442"),
    "492": ("492",),
    "559": ("559", "560"),
    "560": ("560", "559"),
    "665": ("665",),
    "704": ("704",),
    "739": ("739", "740"),
    "740": ("740", "739"),
    "780": ("780", "783"),
    "783": ("783", "780"),
    "833": ("833",),
    "864": ("864", "865"),
    "865": ("865", "864"),
    "1610": ("1610", "1614"),
    "1614": ("1614", "1610"),
    "2186": ("2186", "2202"),
    "2202": ("2202", "2186"),
}


def normalize_rgb_image(img):
    img = img.astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def normalize_multichannel_image(img):
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    flat_img = img.reshape(-1, img.shape[-1])
    mean = flat_img.mean(axis=0, keepdims=True)
    std = flat_img.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (img - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def build_spatial_transform(img_size=None, train=False, use_color_aug=False):
    transforms = []
    if img_size is not None:
        transforms.append(albu.Resize(height=img_size[0], width=img_size[1]))
    if train:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ]
        )
        if use_color_aug:
            transforms.append(
                albu.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.25,
                )
            )
    return albu.Compose(transforms) if transforms else None


def apply_transform(img, mask, spatial_transform, normalize_fn):
    img, mask = np.array(img), np.array(mask, dtype=np.uint8)
    if spatial_transform is not None:
        aug = spatial_transform(image=img.copy(), mask=mask.copy())
        img, mask = aug["image"], aug["mask"]
    img = normalize_fn(img)
    return img, mask


def train_aug(img, mask, img_size=INPUT_IMG_SIZE):
    return apply_transform(
        img,
        mask,
        build_spatial_transform(img_size=img_size, train=True, use_color_aug=True),
        normalize_rgb_image,
    )


def val_aug(img, mask, img_size=TEST_IMG_SIZE):
    return apply_transform(
        img,
        mask,
        build_spatial_transform(img_size=img_size, train=False),
        normalize_rgb_image,
    )


def test_aug(img, mask, img_size=TEST_IMG_SIZE):
    return val_aug(img, mask, img_size=img_size)


def train_aug_rhorc(img, mask, img_size=INPUT_IMG_SIZE):
    return apply_transform(
        img,
        mask,
        build_spatial_transform(img_size=img_size, train=True, use_color_aug=False),
        normalize_multichannel_image,
    )


def val_aug_rhorc(img, mask, img_size=TEST_IMG_SIZE):
    return apply_transform(
        img,
        mask,
        build_spatial_transform(img_size=img_size, train=False),
        normalize_multichannel_image,
    )


def test_aug_rhorc(img, mask, img_size=TEST_IMG_SIZE):
    return val_aug_rhorc(img, mask, img_size=img_size)


def train_aug_rhorc_native(img, mask):
    return apply_transform(
        img,
        mask,
        build_spatial_transform(img_size=None, train=True, use_color_aug=False),
        normalize_multichannel_image,
    )


def val_aug_rhorc_native(img, mask):
    return apply_transform(
        img,
        mask,
        build_spatial_transform(img_size=None, train=False),
        normalize_multichannel_image,
    )


def test_aug_rhorc_native(img, mask):
    return val_aug_rhorc_native(img, mask)


class MADOSDataset(Dataset):
    def __init__(
        self,
        data_root="data/MADOS",
        mode="train",
        resolution="10",
        img_suffix=".png",
        mask_suffix=".tif",
        img_tag="rgb",
        mask_tag="cl",
        transform=test_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
        ignore_index=IGNORE_INDEX,
        rhorc_bands=RHORC_BANDS,
        rhorc_band_specs=MADOS_MULTISPECTRAL_BAND_SPECS,
    ):
        self.data_root = data_root
        self.mode = mode
        self.resolution = str(resolution)
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.img_tag = img_tag
        self.mask_tag = mask_tag
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.ignore_index = ignore_index
        self.rhorc_bands = tuple(str(band) for band in rhorc_bands)
        self.rhorc_band_specs = tuple(
            (str(band), str(resolution)) for band, resolution in rhorc_band_specs
        )
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
        split_path = osp.join(self.data_root, "splits", f"{self.mode}_X.txt")
        with open(split_path, "r", encoding="utf-8") as split_file:
            img_ids = [line.strip() for line in split_file if line.strip()]
        if not img_ids:
            raise ValueError(f"No samples found in split file: {split_path}")
        return img_ids

    def build_sample_path(self, img_id, file_tag, file_suffix):
        scene_name, crop_id = img_id.rsplit("_", 1)
        file_name = f"{scene_name}_L2R_{file_tag}_{crop_id}{file_suffix}"
        return osp.join(self.data_root, scene_name, self.resolution, file_name)

    def build_rhorc_sample_path(self, img_id, band, resolution=None):
        scene_name, crop_id = img_id.rsplit("_", 1)
        file_name = f"{scene_name}_L2R_rhorc_{band}_{crop_id}.tif"
        resolution = self.resolution if resolution is None else str(resolution)
        return osp.join(self.data_root, scene_name, resolution, file_name)

    def resolve_rhorc_band_path(self, img_id, band, resolution=None):
        for candidate_band in RHORC_BAND_ALIASES.get(str(band), (str(band),)):
            candidate_path = self.build_rhorc_sample_path(
                img_id, candidate_band, resolution=resolution
            )
            if osp.exists(candidate_path):
                return candidate_path
        raise FileNotFoundError(
            f"MADOS rhorc band not found for {img_id}: band={band}, resolution={resolution or self.resolution}"
        )

    def load_rhorc_band(self, img_id, band, resolution=None, target_size=None):
        band_path = self.resolve_rhorc_band_path(
            img_id, band, resolution=resolution
        )
        band_image = Image.open(band_path)
        if target_size is not None and band_image.size != target_size:
            band_image = band_image.resize(target_size, Image.BILINEAR)
        band_array = np.array(band_image, dtype=np.float32)
        return np.nan_to_num(band_array, nan=0.0, posinf=0.0, neginf=0.0)

    def load_multires_rhorc(self, img_id):
        first_band, first_resolution = self.rhorc_band_specs[0]
        first_path = self.resolve_rhorc_band_path(
            img_id, first_band, resolution=first_resolution
        )
        with Image.open(first_path) as first_image:
            target_size = first_image.size

        ten_meter_mask = self.build_sample_path(img_id, self.mask_tag, self.mask_suffix)
        if osp.exists(ten_meter_mask):
            with Image.open(ten_meter_mask) as mask_image:
                target_size = mask_image.size

        band_images = [
            self.load_rhorc_band(
                img_id,
                band,
                resolution=band_resolution,
                target_size=target_size,
            )
            for band, band_resolution in self.rhorc_band_specs
        ]
        return np.stack(band_images, axis=-1)

    def load_img(self, img_id):
        if self.img_tag == "rhorc":
            band_images = []
            for band in self.rhorc_bands:
                band_images.append(self.load_rhorc_band(img_id, band))
            return np.stack(band_images, axis=-1)

        if self.img_tag in ("rhorc_multires", "rhorc11", "multispectral"):
            return self.load_multires_rhorc(img_id)

        img_name = self.build_sample_path(img_id, self.img_tag, self.img_suffix)
        if not osp.exists(img_name):
            raise FileNotFoundError(f"MADOS image not found: {img_name}")
        return np.array(Image.open(img_name).convert("RGB"), dtype=np.uint8)

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        mask_name = self.build_sample_path(img_id, self.mask_tag, self.mask_suffix)

        if not osp.exists(mask_name):
            raise FileNotFoundError(f"MADOS mask not found: {mask_name}")

        img = self.load_img(img_id)
        mask = np.array(Image.open(mask_name), dtype=np.uint8)
        mask = np.where(mask > 0, mask - 1, self.ignore_index).astype(np.uint8)
        return img, mask
