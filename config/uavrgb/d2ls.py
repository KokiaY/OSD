import torch
from torch.utils.data import DataLoader
from network.losses import *
from network.datasets.uavrgb_dataset_legacy import *
from network.models.d2ls import DynamicDictionaryLearning 
# Catalyst removed; using plain AdamW

# training hparam
max_epoch = 30  
ignore_index = 255
train_batch_size = 2
val_batch_size = 2
lr = 1e-4
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES) # 4
token_length = num_classes 
classes = CLASSES 
metric_include_indices = (1, 2, 3)

weights_name = "uavrgb_OSD_swinv2_base"
weights_path = "checkpoints/uavrgb/{}".format(weights_name)
test_weights_name = weights_name  # 指定使用的 ckpt文件（不含 .ckpt）
# test_weights_name = "uavrgb_OSD_convnext_base-v1"
log_name = 'uavrgb/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [0]  # 使用GPU
resume_ckpt_path = None
strategy = "auto"

# 启用对比损失 + 设定权重lambda
has_contrastive_loss = True
contrastive_lambda = 0.1
contrastive_loss_weight = contrastive_lambda

# define the network
net = DynamicDictionaryLearning(
    model="swinv2_base",
    token_length=token_length,
    l=3,             #Interactor 使用这个 l 来决定创建多少层 InteractorBlock  交互层数
    has_contrastive_loss=has_contrastive_loss,
    # use_hdpa=False,
    # hdpa_rep_scale=2,
    has_aggregator=False,  # 关闭 HFA
    only_static=False,      # 关闭 EPPG (只用静态 Query)
)

# define the loss
# 切换为DeltaLoss（SoftCE + WeightedBoundaryDice）
loss = DeltaLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader
train_dataset = UAVRGBDataset(
    data_root='data/UAVRGB/Train',
    img_dir='images',
    mask_dir='masks',
    img_suffix='.jpg',
    mask_suffix='.png',
    mode='train',
    mosaic_ratio=0.0,
    transform=train_aug,
    img_size=(512, 960)
)

val_dataset = UAVRGBDataset(
    data_root='data/UAVRGB/Val',
    img_dir='images',
    mask_dir='masks',
    img_suffix='.jpg',
    mask_suffix='.png',
    mode='test',
    mosaic_ratio=0.0,
    transform=val_aug,
    img_size=(512, 960)
)

# optional test dataset for external script use
class UAVRGBTestDataset(UAVRGBDataset):
    def __init__(self, data_root='data/UAVRGB/Test', **kwargs):
        super().__init__(data_root=data_root, mode='test', **kwargs)


test_dataset = UAVRGBTestDataset(
    img_dir='images',
    mask_dir='masks',
    img_suffix='.jpg',
    mask_suffix='.png',
    mosaic_ratio=0.0,
    transform=test_aug,
    img_size=(512, 960)
)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=0,
                          pin_memory=True,
                          persistent_workers=False,
                          prefetch_factor=None,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=uavrgb_collate_fn)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=False,
                        prefetch_factor=None,
                        drop_last=False,
                        collate_fn=uavrgb_collate_fn)

# define the optimizer
base_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = base_optimizer
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
