import torch

from config.mados.d2ls_swin_weighted_v4 import *


prototype_cls_weight = 0.0
prototype_diversity_weight = 0.0

weights_name = "d2ls_swinv2_base_weighted_v4_mpd_reg_none"
weights_path = "checkpoints/mados/{}".format(weights_name)
test_weights_name = weights_name
log_name = "mados/{}".format(weights_name)

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

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-6
)
