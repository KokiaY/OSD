from d2ls_swin_weighted_v4_mados_recipe import *


contrastive_loss_weight = 0.1

weights_name = "d2ls_swinv2_base_weighted_v4_mpd_ms11_mados_recipe_ctr01"
weights_path = "checkpoints/mados/{}".format(weights_name)
test_weights_name = weights_name
log_name = "mados/{}".format(weights_name)
