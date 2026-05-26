from functools import partial

import torch
from torch.utils.data import DataLoader

from network.datasets.m4d_dataset import *
from network.losses import *
from network.models.d2ls import DynamicDictionaryLearning


def build_config(
    *,
    weights_name,
    ablation_group,
    ablation_variant,
    prototypes_per_class=2,
    interaction_layers=3,
    has_aggregator=True,
    has_interactor=True,
    only_static=False,
    has_contrastive_loss=True,
    use_aux_loss=True,
    prototype_aggregation="logsumexp",
    prototype_temperature=1.0,
    prototype_cls_weight=1.0,
    prototype_diversity_weight=0.1,
):
    if only_static:
        has_contrastive_loss = False
        use_aux_loss = False

    max_epoch = 50
    ignore_index = 255
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
    input_img_size = (512, 1024)
    test_img_size = (512, 1024)

    weights_path = "checkpoints/m4d/{}".format(weights_name)
    test_weights_name = weights_name
    log_name = "m4d/ablations/{}".format(weights_name)
    monitor = "val_mIoU"
    monitor_mode = "max"
    save_top_k = 1
    save_last = False
    check_val_every_n_epoch = 1
    pretrained_ckpt_path = None
    gpus = [0]
    resume_ckpt_path = None
    strategy = None

    net = DynamicDictionaryLearning(
        model="swinv2_base",
        token_length=token_length,
        l=interaction_layers,
        pretrained_backbone=True,
        has_aggregator=has_aggregator,
        has_interactor=has_interactor,
        only_static=only_static,
        has_contrastive_loss=has_contrastive_loss,
        prototypes_per_class=prototypes_per_class,
        prototype_aggregation=prototype_aggregation,
        prototype_temperature=prototype_temperature,
        prototype_cls_weight=prototype_cls_weight,
        prototype_diversity_weight=prototype_diversity_weight,
    )

    loss = UnetFormerLoss(ignore_index=ignore_index)

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

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=1e-6
    )

    return {
        name: value
        for name, value in locals().items()
        if name not in {"name", "value"}
    }
