from _d2ls_swin_v4_ablation_base import build_config


globals().update(
    build_config(
        weights_name="m4d_abl_interaction_l1_swinv2_k2",
        ablation_group="interaction depth",
        ablation_variant="L=1",
        interaction_layers=1,
    )
)
