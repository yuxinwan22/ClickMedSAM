# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse
import cc3d
import wandb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)

class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        self.gt_path_files = [
            file for file in self.gt_path_files
            if isfile(join(self.img_path, basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        num_clicks = 10
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_resize = self.resize_longest_side(img_3c)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt_padded = self.pad_image(gt) # (256, 256)
        gt = gt_padded.copy()
        label_ids = np.unique(gt)[1:]
        try:
            label = random.choice(label_ids)
            gt2D = np.uint8(gt == label) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        clicks_coords = torch.stack([torch.tensor(y_indices, dtype=torch.int64), torch.tensor(x_indices, dtype=torch.int64)], dim=1)
        clicks_coords = clicks_coords[torch.randperm(clicks_coords.shape[0])[:num_clicks]]
        
        # num_clicks = min(num_clicks, clicks_coords.shape[0])
        # clicks_coords = clicks_coords[torch.randperm(clicks_coords.shape[0])[:num_clicks]]
        
        # idx_random = np.random.choice(len(y_indices), num_clicks, replace=False)
        # y_indices_clicks = y_indices[idx_random]
        # x_indices_clicks = x_indices[idx_random]
        # max_lcc = 0
        # lcc_mask = None
        # sum_lcc_tmp = None
        
        # # clicks = []
        # for i in range(num_clicks):
        #     # clicks.append(np.array([y_indices_clicks[i], x_indices_clicks[i]]))
        #     p_click = gt_padded[y_indices_clicks[i], x_indices_clicks[i]]
        #     p_threshold = 0.1
        #     p_up = p_click * (1 + p_threshold)
        #     p_down = p_click * (1 - p_threshold)
        #     gt_click = gt_padded.copy()
        #     gt_click[gt_click > p_up] = 0
        #     gt_click[gt_click < p_down] = 0
        #     gt_click[gt_click > 0] = label
        #     connected_components = cc3d.connected_components(gt_click)
        #     sum_cc = np.sum(connected_components)
        #     largest_component_label = np.argmax(np.bincount(connected_components.flat)[1:]) + 1
        #     largest_component_mask = (connected_components==largest_component_label).astype(np.uint8) * 255
        #     sum_lcc = np.sum(largest_component_mask) / 255
        #     # only for debug
        #     if sum_lcc_tmp is None:
        #         sum_lcc_tmp = sum_lcc
        #     else:
        #         if sum_lcc != sum_lcc_tmp:
        #             print("Multi-clicks in gt make sense!")
        #             print(img_name, 'sum_lcc', sum_lcc, 'sum_lcc_tmp', sum_lcc_tmp)
        #             sum_lcc_tmp = sum_lcc

        #     if sum_lcc > max_lcc:
        #         max_lcc = sum_lcc
        #         lcc_mask = largest_component_mask
        #         best_click = np.array([y_indices_clicks[i], x_indices_clicks[i]])
        #     # only for debug
        #     if int(sum_cc) < int(sum_lcc):
        #         print(img_name, 'sum_cc', sum_cc, 'sum_lcc', sum_lcc)
        #         print("Warnings! cc3d may get some error! But still let codes run!")
        # # clicks = np.array(clicks)
        # # clicks = torch.tensor(clicks, dtype=torch.float32)
        # # click_coords = torch.tensor(best_click, dtype=torch.float32).unsqueeze(0)
        click_labels = torch.tensor(label, dtype=torch.float32).repeat(clicks_coords.shape[0])
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long(),
            # "largest_cc_mask": torch.tensor(lcc_mask[None, None, ...]).long(),
            "clicks": (clicks_coords, click_labels)
        }

    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded


# %%
class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image, click, mask=None):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            boxes=None,
            points=click,
            masks=mask,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

def cal_loss(pred, gt, seg, ce, iou):
    logits_pred, iou_pred = pred
    seg_loss, seg_loss_weight = seg
    ce_loss, ce_loss_weight = ce
    iou_loss, iou_loss_weight = iou
    
    l_seg = seg_loss(logits_pred, gt)
    l_ce = ce_loss(logits_pred, gt.float())
    #mask_loss = l_seg + l_ce
    mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
    iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt.bool())
    l_iou = iou_loss(iou_pred, iou_gt)
    #loss = mask_loss + l_iou
    try:
        loss = mask_loss + iou_loss_weight * l_iou
    except:
        l_iou = l_iou.unsqueeze(-1).unsqueeze(-1)
        loss = mask_loss + iou_loss_weight * l_iou
    return loss

def cal_loss_click(pred, gt, seg, ce, iou):
    logits_pred, iou_pred = pred
    seg_loss, seg_loss_weight = seg
    ce_loss, ce_loss_weight = ce
    iou_loss, iou_loss_weight = iou
    
    l_seg = seg_loss(logits_pred, gt)
    l_ce = ce_loss(logits_pred, gt.float())
    mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
    iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt.bool())
    l_iou = iou_loss(iou_pred, iou_gt)
    l_iou = l_iou.unsqueeze(-1).unsqueeze(-1)
    loss = mask_loss + iou_loss_weight * l_iou
    return loss

def reselect_click(bs_size, clicks, logits_pred, click):
    num_clicks = clicks[0].size(1)
    clicks_value = logits_pred[torch.arange(bs_size).view(-1,1), 0, clicks[0][..., 0], clicks[0][..., 1]]
    presence = clicks_value > 0.5

    false_indices = (presence == False).nonzero(as_tuple=True)

    random_indices = torch.randint(0, num_clicks, (bs_size,))

    selected_indices = []
    for i in range(bs_size):
        false_indices_batch = false_indices[1][false_indices[0] == i]
        if len(false_indices_batch) > 0:
            random_idx = false_indices_batch[torch.randint(len(false_indices_batch), (1,))]
        else:
            random_idx = random_indices[i].unsqueeze(0)
        selected_indices.append(clicks[0][i, random_idx].unsqueeze(0))

    selected_indices = torch.cat(selected_indices, dim=0)
    click_coords = torch.cat([selected_indices, click[0]], dim=1)
    label = clicks[1][:, :click_coords.shape[1]]
    return (click_coords, label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_root", type=str, default="./data/npy/CT_Abd",
        help="Path to the npy data root."
    )
    parser.add_argument(
        "-pretrained_checkpoint", type=str, default="lite_medsam.pth",
        help="Path to the pretrained Lite-MedSAM checkpoint."
    )
    parser.add_argument(
        "-resume", type=str, default='workdir/medsam_lite_latest.pth',
        help="Path to the checkpoint to continue training."
    )
    parser.add_argument(
        "-work_dir", type=str, default="./workdir",
        help="Path to the working directory where checkpoints and logs will be saved."
    )
    parser.add_argument(
        "-num_epochs", type=int, default=50,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "-batch_size", type=int, default=1,
        help="Batch size."
    )
    parser.add_argument(
        "-num_workers", type=int, default=8,
        help="Number of workers for dataloader."
    )
    parser.add_argument(
        "-device", type=str, default="cuda:0",
        help="Device to train on."
    )
    parser.add_argument(
        "-bbox_shift", type=int, default=5,
        help="Perturbation to bounding box coordinates during training."
    )
    parser.add_argument(
        "-lr", type=float, default=0.00005,
        help="Learning rate."
    )
    parser.add_argument(
        "-weight_decay", type=float, default=0.01,
        help="Weight decay."
    )
    parser.add_argument(
        "-iou_loss_weight", type=float, default=1.0,
        help="Weight of IoU loss."
    )
    parser.add_argument(
        "-seg_loss_weight", type=float, default=1.0,
        help="Weight of segmentation loss."
    )
    parser.add_argument(
        "-ce_loss_weight", type=float, default=1.0,
        help="Weight of cross entropy loss."
    )
    parser.add_argument(
        "--sanity_check", action="store_true",
        help="Whether to do sanity check for dataloading."
    )
    
    
    parser.add_argument(
        "-wandb_enable", type=bool, default=True,
        help=""
    )
    
    parser.add_argument(
        "-wandb_entity", type=str, default="CVHCI_p24gF_ClickMedSAM",
        help="the place to save your runs. can be your wandb username or team name"
    )
    
    parser.add_argument(
        "-wandb_project", type=str, default="wan_test",
        help="Name of WandB project"
    )
    
    parser.add_argument(
        "-wandb_name", type=str, default="debug",
        help="wandb run name"
    )
    
    parser.add_argument(
        "-wandb_api_key", type=str, default="3fbca40760ef6b876bd8b91911d1008dad6a7b09",
        help="wandb api key"
    )

    args = parser.parse_args()
    # %%
    work_dir = args.work_dir
    data_root = args.data_root
    medsam_lite_checkpoint = args.pretrained_checkpoint
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    bbox_shift = args.bbox_shift
    lr = args.lr
    weight_decay = args.weight_decay
    iou_loss_weight = args.iou_loss_weight
    seg_loss_weight = args.seg_loss_weight
    ce_loss_weight = args.ce_loss_weight
    do_sancheck = args.sanity_check
    checkpoint = args.resume
    
    wandb_enable = args.wandb_enable
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    wandb_name = args.wandb_name
    wandb_api_key = args.wandb_api_key
    
    config = {
        "medsam_lite_checkpoint": medsam_lite_checkpoint,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "bbox_shift": bbox_shift,
        "lr": lr,
        "weight_decay": weight_decay,
        "iou_loss_weight": iou_loss_weight,
        "seg_loss_weight": seg_loss_weight,
        "ce_loss_weight": ce_loss_weight,
    }

    makedirs(work_dir, exist_ok=True)
    if wandb_enable:
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, entity=wandb_entity, config=config, name=wandb_name)

    # %%
    torch.cuda.empty_cache()
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
    
    #%% sanity test of dataset class
    if do_sancheck:
        tr_dataset = NpyDataset(data_root, data_aug=True)
        tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
        for step, batch in enumerate(tr_dataloader):
            # show the example
            _, axs = plt.subplots(1, 2, figsize=(10, 10))
            idx = random.randint(0, 4)

            image = batch["image"]
            gt = batch["gt2D"]
            bboxes = batch["bboxes"]
            names_temp = batch["image_name"]

            axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
            show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
            show_box(bboxes[idx].numpy().squeeze(), axs[0])
            axs[0].axis('off')
            # set title
            axs[0].set_title(names_temp[idx])
            idx = random.randint(4, 7)
            axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
            show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
            show_box(bboxes[idx].numpy().squeeze(), axs[1])
            axs[1].axis('off')
            # set title
            axs[1].set_title(names_temp[idx])
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.savefig(
                join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
                bbox_inches='tight',
                dpi=300
            )
            plt.close()
            break
    # %%
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )

    if medsam_lite_checkpoint is not None:
        if isfile(medsam_lite_checkpoint):
            print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
            medsam_lite_ckpt = torch.load(
                medsam_lite_checkpoint,
                map_location="cpu"
            )
            medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
        else:
            print(f"Pretained weights {medsam_lite_checkpoint} not found, training from scratch")

    medsam_lite_model = medsam_lite_model.to(device)
    medsam_lite_model.train()

    # %%
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # %%
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=5,
        cooldown=0
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')
    seg_loss_batch = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='none')
    ce_loss_batch = nn.BCEWithLogitsLoss(reduction='none')
    iou_loss_batch = nn.MSELoss(reduction='none')
    # %%
    train_dataset = NpyDataset(data_root=data_root, data_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    if checkpoint and isfile(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        checkpoint = torch.load(checkpoint)
        medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10
    # %%
    train_losses = []
    for epoch in range(start_epoch + 1, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            clicks = batch["clicks"]
            clicks = (clicks[0].to(device), clicks[1].to(device))
            bs_size = clicks[0].size(0)
            random_index = torch.randint(0, clicks[0].size(1), (1,))
            click = (clicks[0][:, random_index, :], clicks[1][:, random_index])
            optimizer.zero_grad()
            image, gt2D, click = image.to(device), gt2D.to(device), (click[0].to(device), click[1].to(device))
            logits_pred = None
            for _ in range(3):
                # if i == 0:
                #     logits_pred, iou_pred = medsam_lite_model(image, click)
                # else:
                logits_pred, iou_pred = medsam_lite_model(image, click, logits_pred)
                # if i != 2:
                #     click = reselect_click(bs_size, clicks, logits_pred, click)
                # else:
                #     del click
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            #mask_loss = l_seg + l_ce
            mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            #loss = mask_loss + l_iou
            loss = mask_loss + iou_loss_weight * l_iou
            loss1 = cal_loss(pred=(logits_pred, iou_pred),gt=gt2D, seg=(seg_loss_batch, seg_loss_weight), ce=(ce_loss_batch, ce_loss_weight), iou=(iou_loss_batch, iou_loss_weight))
            epoch_loss[step] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

        epoch_end_time = time()
        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        wandb.log({"train_loss": epoch_loss_reduced})
        train_losses.append(epoch_loss_reduced)
        lr_scheduler.step(epoch_loss_reduced)
        model_weights = medsam_lite_model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss_reduced,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
        if epoch_loss_reduced < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))

        epoch_loss_reduced = 1e10
        # %% plot loss
        plt.plot(train_losses)
        plt.title("Dice + Binary Cross Entropy + IoU Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(work_dir, "train_loss.png"))
        plt.close()
    wandb.finish(quiet=True)
if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()
    main()