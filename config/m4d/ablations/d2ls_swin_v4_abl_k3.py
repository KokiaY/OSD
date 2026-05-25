from _d2ls_swin_v4_ablation_base import build_config


globals().update(
    build_config(
        weights_name="m4d_abl_proto_k3_swinv2_l3",
        ablation_group="sub-prototype number",
        ablation_variant="K=3",
        prototypes_per_class=3,
    )
)
