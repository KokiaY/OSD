from d2ls_swin_weighted_v4 import *


max_epoch = 80
train_batch_size = 5
val_batch_size = 5
lr = 1e-4
weight_decay = 0.01
use_ema = True
ema_decay = 0.999

weights_name = "d2ls_swinv2_base_weighted_v4_mpd_ms11_hybrid_ema"
weights_path = "checkpoints/mados/{}".format(weights_name)
test_weights_name = weights_name
log_name = "mados/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 5
save_last = True

loss = UnetFormerLoss(ignore_index=ignore_index, class_weights=class_weights)

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

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
