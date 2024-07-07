#!/bin/bash
python train_multi_gpus.py -wandb_name click_mask_original -prompt_mode click_mask
python train_multi_gpus.py -wandb_name click_mask_lr0.001 -prompt_mode click_mask -lr 0.001
python train_multi_gpus.py -wandb_name click_mask_lr0.00001 -prompt_mode click_mask -lr 0.00001
python train_multi_gpus.py -wandb_name click_mask_weightdecay0.1 -prompt_mode click_mask -weight_decay 0.1
python train_multi_gpus.py -wandb_name click_mask_weightdecay0.001 -prompt_mode click_mask -weight_decay 0.001
