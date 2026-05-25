from _d2ls_swin_v4_ablation_base import build_config


globals().update(
    build_config(
        weights_name="m4d_abl_full_swinv2_k2_l3",
        ablation_group="component",
        ablation_variant="full",
    )
)
