# EndoVis2018
MICCAI challenge for EndoVis2018

# Instruction for Tensorboardx

pip install tensorboardX
pip install tensorflow

start tensorboard by "tensorboard --logdir=<dir_to_store_log_file>"

# Final Report
See [pdf](materials/15_ZhaoshuoLi_HaoDing_MingyiZheng_final_report) for more details.

# Demo video
[Click here for the demo video](https://youtu.be/EztXBY7mhCk)

# Result

### Mini-Data (20% uniform subsampling)
|Network| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|UNet	|0.87	|0.86	|0.73	|0.76	|0.82	|0.84	|0.68	|0.85	|0.00	|0.66	|0.88	|0.58	|0.71|
|AlbuNet	|0.92	|0.91	|0.80	|0.79	|0.90	|0.90	|0.68	|0.78	|0.00	|0.76	|0.91	|0.71	|0.76|
|AlbuNet+SuperLabel	|0.93	|0.93	|0.82	|0.80	|0.91	|0.90	|0.62	|0.86	|0.00	|0.78	|0.92	|0.77	|0.77|
|DeepLabV3+	|0.91	|0.93	|0.81	|0.82	|0.94	|0.87	|0.51	|0.60	|0.00	|0.76	|0.92	|0.73	|0.73|
|DeepLabv3+SuperLabel	|0.93	|0.93	|0.83	|0.79	|0.91	|0.90	|0.64	|0.85	|0.00	|0.79	|0.92	|0.82	|0.78|
|DeepLabV3+Aug | 0.90 	|0.94	|0.80	|0.84	|0.94	|0.84	|0.53	|0.68	|0.00	|0.59	|0.81	|0.81	|0.72|
|DeepLabv3+SuperLabel+Aug	|0.94	|0.93	|0.83	|0.81	|0.92	|0.92	|0.64	|0.84	|0.00	|0.81	|0.94	|0.83	|0.78|

### Sequence sample
|Network| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|UNet	|0.66	|0.87	|0.76	|0.77	|0.41	|0.22	|0.35	|0.22	|0.00	|0.09	|0.53	|0.00	|0.41	|0.6294|
|AlbuNet	|0.69	|0.90	|0.76	|0.78	|0.51	|0.29	|0.38	|0.15	|0.00	|0.23	|0.59	|0.01	|0.44	|0.6049|
|AlbuNet+SuperLabel	|0.75	|0.94	|0.79	|0.84	|0.60	|0.43	|0.43	|0.45	|0.00	|0.45	|0.62	|0.00	|0.53	|0.5455|
|DeepLabV3+	|0.74	|0.89	|0.76	|0.80	|0.65	|0.29	|0.30	|0.40	|0.00	|0.06	|0.56	|0.00	|0.45	|0.5975|
|DeepLabv3+SuperLabel	|0.74	|0.92	|0.78	|0.83	|0.64	|0.33	|0.33	|0.39	|0.00	|0.20	|0.59	|0.00	|0.48	|0.6048|

### Random sample
|Network| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|AlbuNet+SuperLabel	|0.96	|0.96	|0.9	|0.88	|0.96	|0.95	|0.72	|0.9|	0	|0.83	|0.96	|0.87	|0.82|
|DeepLabV3+	|0.96	|0.95	|0.89	|0.87	|0.96	|0.96	|0.69	|0.9	|0.37	|0.84	|0.97	|0.86	|0.85|
|DeepLabv3+SuperLabel	|0.97	|0.96	|0.89	|0.87	|0.96	|0.96	|0.7	|0.9	|0.38	|0.82	|0.96	|0.89	|0.86|

### Visual comparison of with/without superlabel
see Comparison.ipynb

# Example Notebook
1. [UNet](code/UNet.ipynb)
2. [AlbuNet](code/albunet.ipynb)
3. [DeepLabV3+](code/Deeplabv3+.ipynb)
4. [AlbuNet SuperLabel](code/super_label_albunet.ipynb)
5. [DeepLabV3+ SuperLabel](code/super_label_deeplab.ipynb)

# Architecture
1. [UNet](code/unet.py)
2. [AlbuNet](code/model_from_ternaus.py)
3. [DeepLabV3+](code/deeplabv3p_resnet.py)
4. [AlbuNet SuperLabel](code/model_from_ternaus.py)
5. [DeepLabV3+ SuperLabel](code/deeplabv3p_resnet_super_label.py)

# Components
1. [model_training.py](code/model_training.py)
2. [dataset.py](code/dataset.py)
3. [dice_loss.py](code/dice_loss.py)
4. etc.
