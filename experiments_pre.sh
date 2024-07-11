#!/bin/bash
python train_multi_gpus.py -wandb_name click_mask_pre_epoch100 -prompt_mode click_mask -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100
python train_multi_gpus.py -wandb_name click_mask_lr0.00001_pre_epoch100 -prompt_mode click_mask -lr 0.00001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100
python train_multi_gpus.py -wandb_name click_mask_lr0.000001_pre_epoch100 -prompt_mode click_mask -lr 0.000001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100

python train_multi_gpus.py -wandb_name litemedsam_pre_epoch100  -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100
python train_multi_gpus.py -wandb_name litemedsam_lr0.00001_pre_epoch100  -lr 0.00001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100
python train_multi_gpus.py -wandb_name litemedsam_lr0.000001_pre_epoch100  -lr 0.000001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100

python train_multi_gpus.py -wandb_name click_mask_weightdecay0.1_pre_epoch100 -prompt_mode click_mask -weight_decay 0.1 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100
python train_multi_gpus.py -wandb_name click_mask_weightdecay0.001_pre_epoch100 -prompt_mode click_mask -weight_decay 0.001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth -num_epochs 100