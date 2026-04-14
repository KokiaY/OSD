from functools import partial

from torch.utils.data import DataLoader

from network.losses import *
from network.datasets.m4d_dataset import *
from network.models.d2ls import DynamicDictionaryLearning

# training hparam
max_epoch = 100
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
input_img_size = (512, 1024)
test_img_size = (512, 1024)
class_weights = [1.0, 6.0, 2.5, 12.0, 2.5]

weights_name = "d2ls_hr_weighted"
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

# define the network
net = DynamicDictionaryLearning(
    model="convnext_base",
    token_length=token_length,
    l=3,
    pretrained_backbone=True,
    has_contrastive_loss=has_contrastive_loss,
)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index, class_weights=class_weights)

use_aux_loss = True

# define the dataloader
train_dataset = M4DDataset(
    data_root="data/M4D",
    mode="train",
    img_dir="images",
    mask_dir="labels_1D",
    transform=partial(train_aug, img_size=input_img_size),
    mosaic_ratio=0.0,
    img_size=input_img_size,
)

val_dataset = M4DDataset(
    data_root="data/M4D",
    mode="test",
    img_dir="images",
    mask_dir="labels_1D",
    transform=partial(val_aug, img_size=test_img_size),
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

# define the optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
