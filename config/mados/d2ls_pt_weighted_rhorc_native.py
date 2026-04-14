from torch.utils.data import DataLoader

from network.datasets.mados_dataset import *
from network.losses import *
from network.models.d2ls import DynamicDictionaryLearning


max_epoch = 120
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
input_channels = len(RHORC_BANDS)
train_class_counts = [2912, 2051, 2898, 1192, 6655, 160471, 301366, 178914, 547, 101333, 79696, 11333, 7912, 1512, 8313]
class_weights = [1.299804, 1.548784, 1.30294, 2.031588, 0.859804, 0.175096, 0.127769, 0.165826, 2.999025, 0.220342, 0.248459, 0.658872, 0.788552, 1.803839, 0.769298]

weights_name = "d2ls_pt_weighted_rhorc_native"
weights_path = "checkpoints/mados/{}".format(weights_name)
test_weights_name = weights_name
log_name = "mados/{}".format(weights_name)
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

net = DynamicDictionaryLearning(
    model="convnext_base",
    token_length=token_length,
    l=3,
    pretrained_backbone=True,
    has_contrastive_loss=has_contrastive_loss,
    input_channels=input_channels,
)

loss = UnetFormerLoss(ignore_index=ignore_index, class_weights=class_weights)

use_aux_loss = True

train_dataset = MADOSDataset(
    data_root="data/MADOS",
    mode="train",
    resolution="10",
    img_suffix=".tif",
    img_tag="rhorc",
    rhorc_bands=RHORC_BANDS,
    transform=train_aug_rhorc_native,
    mosaic_ratio=0.0,
    img_size=ORIGIN_IMG_SIZE,
)

val_dataset = MADOSDataset(
    data_root="data/MADOS",
    mode="val",
    resolution="10",
    img_suffix=".tif",
    img_tag="rhorc",
    rhorc_bands=RHORC_BANDS,
    transform=val_aug_rhorc_native,
    mosaic_ratio=0.0,
    img_size=ORIGIN_IMG_SIZE,
)

test_dataset = MADOSDataset(
    data_root="data/MADOS",
    mode="test",
    resolution="10",
    img_suffix=".tif",
    img_tag="rhorc",
    rhorc_bands=RHORC_BANDS,
    transform=test_aug_rhorc_native,
    mosaic_ratio=0.0,
    img_size=ORIGIN_IMG_SIZE,
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
