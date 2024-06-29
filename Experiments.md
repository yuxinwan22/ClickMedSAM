# Multi-GPU
To fine-tune Lite-MedSAM on multiple GPUs, run:
```bash
python train_multi_gpus.py \
    -i data/npy \ ## path to the training dataset \
    -v data/npy/CT_Abd \ ## path to the validation dataset \
    -wandb_enable True \ ## enable wandb \
    -wandb_entity CVHCI_p24gF_ClickMedSAM \ ## the place to save your runs. Please don't change in fine tuning \
    -wandb_project wan_test \ ## Name of the whole experiments set \
    -wandb_name  \ ## name of the current wandb run name \
    -wandb_api_key \ ## please enter your own wanb api key here \
    -prompt_mode click_mask \ ## You can choose bbox, click_mask or click_re, please don't use click_re now! \
    -num_clicks 10 \ ## number of candidate clicks, only available when prompt mode is click_re! \
    -num_reselect 3 \ ## number of reselected clicks, only available when prompt mode is click_re! \
    -task_name MedSAM-Lite-Box \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir ./work_dir_ddp \
    -batch_size 4 \ ## batch size per GPU \
    -which_gpus [0,1,2,3] \ ## select available GPUs, and make sure the whole batch size is 14 \
    -num_workers 8 \
    -num_epochs 50 \
    -lr 0.0005 \
    --data_aug \ ## use data augmentation
    -iou_loss_weight 1.0 \ ## Weight of IoU loss \
    -seg_loss_weight 1.0 \ ## Weight of segmentation loss \
    -ce_loss_weight 1.0 \ ## Weight of cross entropy loss \
    -world_size <WORLD_SIZE> \ ## Total number of GPUs will be used
    -node_rank 0 \ ## if training on a single machine, set to 0
    -init_method tcp://<MASTER_ADDR>:<MASTER_PORT>
```

