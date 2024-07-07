#!/bin/bash
python train_multi_gpus.py -wandb_name click_mask_pre -prompt_mode click_mask -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth
python train_multi_gpus.py -wandb_name click_mask_lr0.00001_pre -prompt_mode click_mask -lr 0.00001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth
python train_multi_gpus.py -wandb_name click_mask_lr0.000001_pre -prompt_mode click_mask -lr 0.000001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth

python train_multi_gpus.py -wandb_name litemedsam_pre  -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth
python train_multi_gpus.py -wandb_name litemedsam_lr0.00001_pre  -lr 0.00001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth
python train_multi_gpus.py -wandb_name litemedsam_lr0.000001_pre  -lr 0.000001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth

python train_multi_gpus.py -wandb_name click_mask_weightdecay0.1_pre -prompt_mode click_mask -weight_decay 0.1 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth
python train_multi_gpus.py -wandb_name click_mask_weightdecay0.001_pre -prompt_mode click_mask -weight_decay 0.001 -pretrained_checkpoint work_dir/LiteMedSAM/lite_medsam.pth