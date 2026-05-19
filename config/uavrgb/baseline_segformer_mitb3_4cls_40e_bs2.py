from functools import partial

from torch.utils.data import DataLoader

from network.datasets.uavrgb_dataset_4class import *
from network.losses import *
from network.models.baselines import HuggingFaceSegFormer


max_epoch = 40
ignore_index = IGNORE_INDEX
train_batch_size = 2
val_batch_size = 2
lr = 6e-5
weight_decay = 0.01
num_workers = 0
num_classes = len(CLASSES)
classes = CLASSES
input_img_size = INPUT_IMG_SIZE
test_img_size = TEST_IMG_SIZE

weights_name = "baseline_segformer_mitb3_4cls_40e_bs2"
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
strategy = None
precision = "16-mixed"
enable_progress_bar = False
has_contrastive_loss = False

net = HuggingFaceSegFormer(
    num_classes=num_classes,
    model_name="nvidia/mit-b3",
    pretrained=True,
    local_files_only=True,
)
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

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
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-6
)
