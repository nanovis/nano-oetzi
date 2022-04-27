# Nano-Otzi: Cryo-Electron Tomography Visualization Guided by Learned Segmentations
This repository contains code for training and inference of volumes with use of deep segmentation model presented in Nano-Oetzi paper.

You need to have `cuda 11.0` installed. You might have problems setting-up the environment with different versions.

The easiest is to set-up a conda virtual environment using `vorecem_env.yaml` file from the repository.
```
conda env create --prefix <path_to_environment> --file=<path_to_conda_env.yaml_file>
```
Once activating the environment:
```
conda activate <path_to_environment>
```
you need to install additional packages using `pip` command:
```
pip install git+https://github.com/aliutkus/torchinterp1d.git@master#egg=torchinterp1d
pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer@master#egg=ranger
```
The environment should be ready to use.

## Overview:
### External tools:
For training and inference the data needs to be converted to torchvtk format. We provide tools to convert the MRC volumes into torchvtk.
```
segmentation/conversion/                 # Script and Docs to convert .mrc to torchvtk format
segmentation/iunets/                     # UNet and iUNet models, from https://github.com/cetmann/iunets
```

### Training and test scripts:
The scripts enable training the model with or without GPU. The model training depends on the dataset size. In our case we used 60 volumes: 50 for training, 5 for validation, and 5 for testing. The model training took 5 days and 16 hours to converge. The fine-tuning of the selected transfer-learned model took an additional 2 days and 15 hours. The volumes resolution was: 1024x1440x\[227-500\]
```
segmentation/train.py                    # Script to start training for foreground - background data. See --help for arguments
segmentation/lmodule.py                  # Script with the Lightning Module. Implements training for foreground - background data procedure
segmentation/test.py                     # Script to start testing for foreground - background data
segmentation/train_gpus.py               # Script to start training for foreground - background data with many GPUs. See --help for arguments
segmentation/lmodule_gpus.py             # Script with the Lightning Module. Implements training on many GPUs for foreground - background data procedure

segmentation/transfer.py                 # Script to start training for many classes data. See --help for arguments
segmentation/lmodule_transfer.py         # Script with the Lightning Module. Implements training for many classes data procedure
segmentation/test_transfer.py            # Script to start testing for many classes data
segmentation/transfer_gpus.py            # Script to start training for many classes data data with many GPUs. See --help for arguments
segmentation/lmodule_transfer_gpus.py    # Script with the Lightning Module. Implements training on many GPUs for many classes data procedure
```

### Support scripts and files
Our volumes were to big to use them for training as a whole. We needed to split them into chunk of resolution 512x512x\[227-500\].
```
segmentation/split.py                       # Script for splitting data into 9 chunks
segmentation/normalize_data.py              # Script for normalizing foreground - background data
segmentation/normalize_data_all_channels.py # Script for normalizing many classes data

segmentation/stitch.py                      # Script for stiching foreground - background data
segmentation/stitch_all_channels.py         # Script for stiching many classes data

conda_env.yaml                              # Docs listing the necessary py packages to run these scripts.
```

## Set up and running scripts
### Split volume
Splits volume into individual chunks of size depth x 512 x 512 voxels, where depth needs to be less than 512 voxels
```
python ./split.py <input_mrc_volume> <output_folder>
```
### Normalize
Checks if split data is correctly bundeled and normalizes the data within chunks for F-B segmentation:
```
python ./normalize_data.py <path_to_folder_with_chunks> <output_folder>
```
or for 4-class segmentation:
```
python ./normalize_data_all_channels.py <path_to_folder_with_chunks> <output_folder>
```

### Foreground-background training
Run train script from `segmentation` folder.
```
train.py <path_to_train_data_folder> --run_id <run_id_for_logging> --max_epochs <number_of_epochs>
```

### 4-class transfer learning
Run transfer script from `segmentation` folder.
```
transfer.py <path_to_train_data_folder> --run_id <run_id_for_logging> --pretrained <path_to_pretrained_foreground-background_checkpoint_file> --max_epochs <number_of_epochs>
```

### Inference
Run script from `segmentation` folder.
```
python test_transfer.py <path_to_test_data_folder> --checkpoint <path_to_checkpiont_file> --run_id <run_id_for_logging>
```

### Stitch chunks
The script stitches prediction chunks into output volumes
```
python ./stitch.py <input_file_name_prefix> <prediction_chunks_folder_path> <output_folder_path> <splits_json_file_path> <tile_locations_file_name_prefix> <output_file_name_prefix>
python ./stitch_all_channels.py <input_file_name_prefix> <prediction_chunks_folder_path> <output_folder_path> <splits_json_file_path> <tile_locations_file_name_prefix> <output_file_name_prefix>
```

## Pretrained models
```
models/foreground_background_model.ckpt: Pretrained model for foreground-background segmentation
models/four_classes_model.ckpt: Pretrained model for four classes segmentation
```
