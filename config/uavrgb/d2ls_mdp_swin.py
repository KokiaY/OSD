import torch
from torch.utils.data import DataLoader

from network.datasets.uavrgb_dataset_legacy import *
from network.losses import *
from network.models.d2ls import DynamicDictionaryLearning


max_epoch = 30
ignore_index = 255
train_batch_size = 2
val_batch_size = 2
lr = 1e-4
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
token_length = num_classes
classes = CLASSES
metric_include_indices = (1, 2, 3)

prototypes_per_class = 2
prototype_aggregation = "logsumexp"
prototype_temperature = 1.0
prototype_cls_weight = 1.0
prototype_diversity_weight = 0.1

weights_name = "uavrgb_OSD_mdp_swinv2_base"
weights_path = "checkpoints/uavrgb/{}".format(weights_name)
test_weights_name = weights_name
log_name = "uavrgb/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [0]
resume_ckpt_path = None
strategy = "auto"

has_contrastive_loss = True
contrastive_lambda = 0.1
contrastive_loss_weight = contrastive_lambda

net = DynamicDictionaryLearning(
    model="swinv2_base",
    token_length=token_length,
    l=3,
    has_contrastive_loss=has_contrastive_loss,
    has_aggregator=False,
    only_static=False,
    prototypes_per_class=prototypes_per_class,
    prototype_aggregation=prototype_aggregation,
    prototype_temperature=prototype_temperature,
    prototype_cls_weight=prototype_cls_weight,
    prototype_diversity_weight=prototype_diversity_weight,
)

loss = DeltaLoss(ignore_index=ignore_index)
use_aux_loss = True

train_dataset = UAVRGBDataset(
    data_root="data/UAVRGB/Train",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".jpg",
    mask_suffix=".png",
    mode="train",
    mosaic_ratio=0.0,
    transform=train_aug,
    img_size=(512, 960),
)

val_dataset = UAVRGBDataset(
    data_root="data/UAVRGB/Val",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".jpg",
    mask_suffix=".png",
    mode="test",
    mosaic_ratio=0.0,
    transform=val_aug,
    img_size=(512, 960),
)


class UAVRGBTestDataset(UAVRGBDataset):
    def __init__(self, data_root="data/UAVRGB/Test", **kwargs):
        super().__init__(data_root=data_root, mode="test", **kwargs)


test_dataset = UAVRGBTestDataset(
    img_dir="images",
    mask_dir="masks",
    img_suffix=".jpg",
    mask_suffix=".png",
    mosaic_ratio=0.0,
    transform=test_aug,
    img_size=(512, 960),
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
    shuffle=True,
    drop_last=True,
    collate_fn=uavrgb_collate_fn,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
    drop_last=False,
    collate_fn=uavrgb_collate_fn,
)

base_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = base_optimizer
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epoch,
    eta_min=1e-6,
)
