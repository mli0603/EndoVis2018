# EndoVis2018
MICCAI challenge for EndoVis2018

# Instruction for Tensorboardx

pip install tensorboardX
pip install tensorflow

start tensorboard by "tensorboard --logdir=<dir_to_store_log_file>"

# Log
2019.05.07 vanilla_trained_unet.py: no data augmentation, pretraining, using Max's dice loss (no per class loss calculation)
dice score: 0.62

2019.05.08 vanilla_trained_unet_new_loss.py: no data augmentation, pretraining, using Hao's dice loss (per class loss calculation, all weights initialized to 1)
dice score: 0.74
