from functools import partial

from torch.utils.data import DataLoader

from network.losses import *
from network.datasets.uavrgb_dataset import *
from network.models.d2ls import DynamicDictionaryLearning


max_epoch = 80
ignore_index = IGNORE_INDEX
train_batch_size = 4
val_batch_size = 4
lr = 1e-4
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_workers = 0
num_classes = len(CLASSES)
token_length = num_classes
classes = CLASSES
input_img_size = INPUT_IMG_SIZE
test_img_size = TEST_IMG_SIZE
prototypes_per_class = 2
prototype_aggregation = "logsumexp"
prototype_temperature = 1.0
prototype_cls_weight = 1.0
prototype_diversity_weight = 0.1

weights_name = "d2ls_swinv2_base_v4_mpd"
weights_path = "checkpoints/uavrgb/{}".format(weights_name)
test_weights_name = weights_name
log_name = "uavrgb/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 5
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [0]
resume_ckpt_path = None
strategy = None
has_contrastive_loss = True

net = DynamicDictionaryLearning(
    model="swinv2_base",
    token_length=token_length,
    l=3,
    pretrained_backbone=True,
    has_contrastive_loss=has_contrastive_loss,
    prototypes_per_class=prototypes_per_class,
    prototype_aggregation=prototype_aggregation,
    prototype_temperature=prototype_temperature,
    prototype_cls_weight=prototype_cls_weight,
    prototype_diversity_weight=prototype_diversity_weight,
)

loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

train_dataset = UAVRGBDataset(
    data_root="data/UAVRGB",
    mode="train",
    transform=partial(train_aug, img_size=input_img_size),
    mosaic_ratio=0.0,
    img_size=input_img_size,
)

val_dataset = UAVRGBDataset(
    data_root="data/UAVRGB",
    mode="val",
    transform=partial(val_aug, img_size=test_img_size),
    mosaic_ratio=0.0,
    img_size=test_img_size,
)

test_dataset = UAVRGBDataset(
    data_root="data/UAVRGB",
    mode="test",
    transform=partial(test_aug, img_size=test_img_size),
    mosaic_ratio=0.0,
    img_size=test_img_size,
)

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
