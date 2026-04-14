import albumentations as albu
import numpy as np
from torch.utils.data import DataLoader

from network.datasets.m4d_dataset import *
from network.losses import *
from network.models.d2ls import DynamicDictionaryLearning


max_epoch = 150
ignore_index = 255
train_batch_size = 2
val_batch_size = 2
lr = 1e-4
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_workers = 0
num_classes = len(CLASSES)
token_length = num_classes
classes = CLASSES
input_img_size = (650, 1250)
test_img_size = (650, 1250)

weights_name = "d2ls_original"
weights_path = "checkpoints/m4d/{}".format(weights_name)
test_weights_name = weights_name
log_name = "m4d/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [0]
resume_ckpt_path = None
strategy = None
has_contrastive_loss = True
contrastive_loss_weight = 1.0


def get_training_transform():
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.25,
            ),
            albu.Normalize(),
        ]
    )


def train_aug_original(img, mask):
    img, mask = np.array(img), np.array(mask, dtype=np.uint8)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    return aug["image"], aug["mask"]


def get_val_transform():
    return albu.Compose([albu.Normalize()])


def val_aug_original(img, mask):
    img, mask = np.array(img), np.array(mask, dtype=np.uint8)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    return aug["image"], aug["mask"]


net = DynamicDictionaryLearning(
    model="convnext_base",
    token_length=token_length,
    l=3,
    pretrained_backbone=True,
    has_contrastive_loss=has_contrastive_loss,
)

loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

train_dataset = M4DDataset(
    data_root="data/M4D",
    mode="train",
    img_dir="images",
    mask_dir="labels_1D",
    transform=train_aug_original,
    mosaic_ratio=0.0,
    img_size=input_img_size,
)

val_dataset = M4DDataset(
    data_root="data/M4D",
    mode="test",
    img_dir="images",
    mask_dir="labels_1D",
    transform=val_aug_original,
    mosaic_ratio=0.0,
    img_size=test_img_size,
)

test_dataset = val_dataset

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
