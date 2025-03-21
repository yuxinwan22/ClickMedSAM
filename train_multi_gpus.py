# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
from shutil import copyfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import multiprocessing as mp
from torch import distributed as dist
from datetime import datetime

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

import wandb
import shutil

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]

colors = [(r/255, g/255, b/255) for r, g, b in colors]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_npy_path', type=str,
                        default='data/npy',
                        help='Path to training npy files; two subfolders: gts and imgs')
    parser.add_argument('-v', '--val_npy_path', type=str,
                        default='data/npy/CT_Abd',
                        help='Path to validating npy files; two subfolders: gts and imgs')
    parser.add_argument('-task_name', type=str, default='click_mask_pre')
    parser.add_argument('-pretrained_checkpoint', type=str, default='/home/ies/wan/ClickMedSAM/work_dir/LiteMedSAM/lite_medsam.pth',
                        help='Path to pretrained MedSAM-Lite checkpoint')
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('--data_aug', action='store_true', default=True,
                        help='use data augmentation during training')
    # train
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-which_gpus', type=list, default=[3,4,5,6],
                        help='Which GPUs will be used')
    parser.add_argument('-num_workers', type=int, default=16)
    
    # Optimizer parameters
    parser.add_argument('-weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('-lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    
    # loss weight
    parser.add_argument("-iou_loss_weight", type=float, default=1.0,
                        help="Weight of IoU loss.")
    parser.add_argument("-seg_loss_weight", type=float, default=1.0,
                        help="Weight of segmentation loss.")
    parser.add_argument("-ce_loss_weight", type=float, default=1.0,
                        help="Weight of cross entropy loss.")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Whether to do sanity check for dataloading.")
    
    ## Distributed training args
    # parser.add_argument('-world_size', type=int, default=1,
    #                     help='world size, Total number of GPUs will be used')
    # parser.add_argument('-which_gpus', type=list, default=[5,6],
    #                     help='Which GPUs will be used')
    parser.add_argument('-node_rank', type=int, default=0,
                        help='Node rank, if training on a single machine, set to 0')
    parser.add_argument('-bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    parser.add_argument('-resume', type = str, default = '', required=False,
                        help="Resuming training from a work_dir, work_dir/click_mask_pre_epoch100-20240708-1052/medsam_lite_best.pth")
    parser.add_argument('-init_method', type = str, default = "env://")
    
    ## wandb
    parser.add_argument("-wandb_enable", type=bool, default=True,
                        help="")
    parser.add_argument("-wandb_entity", type=str, default="CVHCI_p24gF_ClickMedSAM",
                        help="the place to save your runs. can be your wandb username or team name")
    parser.add_argument("-wandb_project", type=str, default="finetune",
                        help="Name of WandB project")
    parser.add_argument("-wandb_name", type=str, default="click_mask_multi_pre",
                        help="wandb run name")
    parser.add_argument("-wandb_api_key", type=str, default="3fbca40760ef6b876bd8b91911d1008dad6a7b09",
                        help="wandb api key")
    
    ## interfaces
    parser.add_argument("-prompt_mode", type=str, default="click_mask",
                        help="bbox, click_re, click_mask")
    parser.add_argument("-click_mode", type=str, default="multi",
                        help="single or multi")
    parser.add_argument("-num_clicks", type=int, default="10",
                        help="number of candidate clicks, only valuable when prompt mode is click_re!")
    parser.add_argument("-num_reselect", type=int, default="3",
                        help="number of reselected clicks, only valuable when prompt mode is click_re!")
    
    
    
    
    args = parser.parse_args()

    return args


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


@torch.no_grad()
def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)


def revert_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
    # Original author: Kapil Yedidi (@kapily)
    converted_module = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        # Unfortunately, SyncBatchNorm does not store the original class - if it did
        # we could return the one that was originally created.
        converted_module = nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                converted_module.weight = module.weight
                converted_module.bias = module.bias
        converted_module.running_mean = module.running_mean
        converted_module.running_var = module.running_var
        converted_module.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            converted_module.qconfig = module.qconfig
    for name, child in module.named_children():
        converted_module.add_module(name, revert_sync_batchnorm(child))
    del module

    return converted_module


class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True, num_clicks=10):
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
        self.num_clicks = num_clicks
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
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
        clicks_coords = clicks_coords[torch.randperm(clicks_coords.shape[0])[:self.num_clicks]]
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
            "image": torch.tensor(img_padded).float(), # torch.Size([3, 256, 256])
            "gt2D": torch.tensor(gt2D[None, :,:]).long(), # torch.Size([1, 256, 256])
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long(),
            "clicks": (clicks_coords, click_labels) # torch.Size([num_clicks, 2]), torch.Size([num_clicks])
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

def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    """
    batch_dict = {}
    for key in batch[0].keys():
        if key == "image_name":
            batch_dict[key] = [sample[key] for sample in batch]
        else:
            batch_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)

    return batch_dict

#%% sanity test of dataset class
def sanity_check_dataset(args):
    print('tr_npy_path', args.tr_npy_path)
    tr_dataset = NpyDataset(args.tr_npy_path, data_aug=args.data_aug)
    print('len(tr_dataset)', len(tr_dataset))
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    makedirs(args.work_dir, exist_ok=True)
    for step, batch in enumerate(tr_dataloader):
        # print(image.shape, gt.shape, bboxes.shape)
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
        # plt.show()  
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(args.work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break

# def visualize_predictions(image, gt2D, logits_pred, click, step, save_path=None):
#     plt.figure(figsize=(10, 5))

#     # 绘制点击点的辅助函数
#     def plot_clicks(coords, labels):
#         for click_idx in range(coords.shape[0]):
#             coord = coords[click_idx]
#             label = labels[click_idx]
#             plt.scatter(coord[1], coord[0], color='blue' if label == 1 else 'blue', marker='x')

#     # 图像带真实掩码和点击点
#     plt.subplot(1, 2, 1)
#     plt.imshow(image.permute(1, 2, 0).cpu().numpy())
#     # 绘制真实掩码
#     plt.imshow(gt2D.squeeze().cpu().numpy(), cmap='viridis', alpha=0.5)  # 叠加真实掩码
#     # 绘制点击点
#     plot_clicks(click[0][0].cpu().numpy(), click[1][0].cpu().numpy())
#     plt.title('Image with Ground Truth Mask and Clicks')
#     plt.axis('off')

#     # 图像带预测掩码和点击点
#     plt.subplot(1, 2, 2)
#     plt.imshow(image.permute(1, 2, 0).cpu().numpy())
#     # 绘制预测掩码
#     pred_mask = logits_pred[0].squeeze().detach().cpu().numpy()
#     pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
#     plt.imshow(pred_mask_bin, cmap='viridis', alpha=0.5)  # 叠加预测掩码
#     # 绘制点击点
#     plot_clicks(click[0][0].cpu().numpy(), click[1][0].cpu().numpy())
#     plt.title('Image with Prediction Mask and Clicks')
#     plt.axis('off')

#     plt.suptitle(f'Step {step}')

#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()

def plot_image_with_mask_and_clicks(image, mask, click, save_path):
    h, w = image.shape[1], image.shape[2]
    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    if click[1][0].shape[0] == 1:
        color_mask = [(0, 0, 0), colors[int(click[1][0])]]
    else:
        color_mask = [(0, 0, 0), colors[int(click[1][0][0])]]
    cm = ListedColormap(color_mask)
    plt.imshow(mask, cmap=cm, alpha=0.5)
    plot_clicks(click[0][0].cpu().numpy(), click[1][0].cpu().numpy())
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def plot_clicks(coords, labels):
    for click_idx in range(coords.shape[0]):
        coord = coords[click_idx]
        label = labels[click_idx]
        plt.scatter(coord[1], coord[0], color='red', marker='x')

def visualize_gt(image, gt2D, click, step, save_path=None):
    gt_mask = gt2D.squeeze().cpu().numpy()
    file_path = f"{save_path}/step_{step}_gt.png" if save_path else f"step_{step}_gt.png"
    plot_image_with_mask_and_clicks(image, gt_mask, click, file_path)

def visualize_pred(image, logits_pred, click, step, layer, save_path=None):
    pred_mask = logits_pred[0].squeeze().detach().cpu().numpy()
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
    file_path = f"{save_path}/step_{step}_layer_{layer}_pred.png" if save_path else f"step_{step}_pred.png"
    plot_image_with_mask_and_clicks(image, pred_mask_bin, click, file_path)


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
        
    def forward(self, image, boxes=None, clicks=None, mask=None):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            boxes=boxes,
            points=clicks,
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

def main(args):
    torch.cuda.empty_cache()
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10086'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.which_gpus).replace(' ', '').replace('[', '').replace(']', '')
    ngpus_per_node = torch.cuda.device_count()
    print("Spawning processes")
    mp.spawn(main_worker, nprocs=len(args.which_gpus), args=(ngpus_per_node, args))


def main_worker(local_rank, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + local_rank
    world_size = len(args.which_gpus)
    print(f"[Rank {rank}]: Use GPU: {args.which_gpus[local_rank]} for training")
    is_main_host = rank == 0
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.wandb_name + "-" + run_id)
    if is_main_host:
        makedirs(model_save_path, exist_ok=True)
        copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:{}".format(local_rank))
    dist.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    wandb_enable = args.wandb_enable
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    wandb_name = args.wandb_name
    wandb_api_key = args.wandb_api_key
    
    if wandb_enable and is_main_host:
        wandb.login(key=wandb_api_key)
        del args.wandb_enable, args.wandb_api_key, args.wandb_project, args.wandb_entity, args.wandb_name
        wandb.init(project=wandb_project, entity=wandb_entity, config=args, name=wandb_name)
    
    
    
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
    
    if (not os.path.exists(args.resume)) and isfile(args.pretrained_checkpoint):
        ## Load pretrained checkpoint if there's no checkpoint to resume from and there's a pretrained checkpoint
        if is_main_host:
            print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
        medsam_lite_checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
        medsam_lite_model.load_state_dict(medsam_lite_checkpoint, strict=True)
    else:
        if is_main_host:
            print(f"Pretained weights {args.pretrained_checkpoint} not found, training from scratch")

    medsam_lite_model = medsam_lite_model.to(device)

    ## Make sure there's only 2d BN layers, so that I can revert them properly
    for module in medsam_lite_model.modules():
        cls_name = module.__class__.__name__
        if "BatchNorm" in cls_name:
            assert cls_name == "BatchNorm2d" 
    medsam_lite_model = nn.SyncBatchNorm.convert_sync_batchnorm(medsam_lite_model)

    medsam_lite_model = nn.parallel.DistributedDataParallel(
        medsam_lite_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb
    )
    medsam_lite_model.train()
    # %%
    if is_main_host:
        print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # %%
    optimizer = optim.AdamW(
        medsam_lite_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
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
    # %%
    train_dataset = NpyDataset(data_root=args.tr_npy_path, data_aug=True, num_clicks=args.num_clicks)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_dataset = NpyDataset(data_root=args.val_npy_path, data_aug=False, num_clicks=args.num_clicks)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler
    )
    # %%

    if os.path.exists(args.resume):
        # ckpt_folders = sorted(listdir(args.resume))
        # ckpt_folders = [f for f in ckpt_folders if (f.startswith(args.task_name) and isfile(join(args.resume, f, 'medsam_lite_latest.pth')))]
        if is_main_host:
            print('*'*20)
        # print('existing ckpts in', args.resume, ckpt_folders)
        # find the latest ckpt folders
        # time_strings = [f.split(args.task_name + '-')[-1] for f in ckpt_folders]
        # dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
        # latest_date = max(dates)
        latest_ckpt = args.resume
        if is_main_host:
            print('Loading from', latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=device)
        medsam_lite_model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        if is_main_host:
            print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10

    train_losses = []
    val_losses = []
    epoch_times = []
    vis_dir = os.path.join(model_save_path, 'visualization')
    vis_train_dir = os.path.join(vis_dir, 'train')
    vis_val_dir = os.path.join(vis_dir, 'validation')
    os.makedirs(vis_train_dir, exist_ok=True)
    os.makedirs(vis_val_dir, exist_ok=True)
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        epoch_loss_val = [1e10 for _ in range(len(val_loader))]
        epoch_start_time = time()
        vis_train_epoch_dir = os.path.join(vis_train_dir, str(epoch))
        vis_val_epoch_dir = os.path.join(vis_val_dir, str(epoch))
        os.makedirs(vis_train_epoch_dir, exist_ok=True)
        os.makedirs(vis_val_epoch_dir, exist_ok=True)
        if is_main_host:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader
        for step, batch in enumerate(pbar):
            image = batch["image"] # torch.Size([B, 3, 256, 256])
            gt2D = batch["gt2D"] # torch.Size([B, 1, 256, 256])
            boxes = batch["bboxes"] # torch.Size([B, 1, 1, 4])
            clicks = batch["clicks"] # coord range 0-255 (torch.Size([B, num_clicks_candidates, 2]), torch.Size([B, num_clicks_candidates]))
            clicks = (clicks[0].to(device), clicks[1].to(device))
            random_index = torch.randint(0, clicks[0].size(1), (1,))
            if args.click_mode == "single":
                click = (clicks[0][:, random_index, :], clicks[1][:, random_index]) # torch.Size([B, 1, 2]), torch.Size([B, 1])
            elif args.click_mode == "multi":
                click = (clicks[0][:, :random_index+1, :], clicks[1][:, :random_index+1])
            else:
                raise ValueError("Invalid click mode! Please enter single or multi")
            optimizer.zero_grad()
            image, gt2D, click, boxes = image.to(device), gt2D.to(device), (click[0].to(device), click[1].to(device)), boxes.to(device)
            
            if args.prompt_mode == "click_re":
                logits_pred = None
                for i in range(args.num_reselect):
                    if i != args.num_reselect - 1:
                        with torch.no_grad():
                            logits_pred_tmp, _ = medsam_lite_model(image, None, click, None)
                            # click = reselect_click(batch_size, clicks, logits_pred_tmp, click)
                    else:
                        del logits_pred_tmp
                        medsam_lite_model.train()
                        logits_pred, iou_pred = medsam_lite_model(image, None, click, None)
                        del click
            elif args.prompt_mode == "click_mask":  
                with torch.no_grad():
                    logits_pred0, _ = medsam_lite_model(image, None, click, None)
                    logits_pred1, _ = medsam_lite_model(image, None, click, logits_pred0)
                medsam_lite_model.train()
                logits_pred, iou_pred = medsam_lite_model(image, None, click, logits_pred1) # torch.Size([B, 1, 256, 256]), torch.Size([B, 1])
                # visualize_predictions(image[0], gt2D[0], logits_pred, click, epoch, save_path=f'visualization_epoch_{epoch}.png')
                if is_main_host and step % 5 == 0:
                    visualize_gt(image[0], gt2D[0], click, step, save_path=vis_train_epoch_dir)
                    visualize_pred(image[0], logits_pred0, click, step, 1, save_path=vis_train_epoch_dir)
                    visualize_pred(image[0], logits_pred1, click, step, 2, save_path=vis_train_epoch_dir)
                    visualize_pred(image[0], logits_pred, click, step, 3, save_path=vis_train_epoch_dir)
                del logits_pred0, logits_pred1
            elif args.prompt_mode == "bbox":
                logits_pred, iou_pred = medsam_lite_model(image, boxes, None, None)
                # visualize_predictions(image[0], gt2D[0], logits_pred, click, epoch, save_path=f'visualization_epoch_{epoch}.png')
            else:
                raise ValueError("Invalid prompt mode! Please enter click_mask or click_re or bbox")

            loss = cal_loss(pred=(logits_pred, iou_pred),gt=gt2D, seg=(seg_loss, args.seg_loss_weight), ce=(ce_loss, args.ce_loss_weight), iou=(iou_loss, args.iou_loss_weight))
            epoch_loss[step] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # pbar.set_description(f"[RANK {rank}] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
            if is_main_host:
                pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
        
        validation = True
        if validation == True:
            with torch.no_grad():
                medsam_lite_model.eval()
                for step, batch in enumerate(val_loader):
                    image = batch["image"]
                    gt2D = batch["gt2D"]
                    boxes = batch["bboxes"]
                    clicks = batch["clicks"]
                    clicks = (clicks[0].to(device), clicks[1].to(device))
                    random_index = torch.randint(0, clicks[0].size(1), (1,))
                    if args.click_mode == "single":
                        click = (clicks[0][:, random_index, :], clicks[1][:, random_index]) # torch.Size([B, 1, 2]), torch.Size([B, 1])
                    elif args.click_mode == "multi":
                        click = (clicks[0][:, :random_index+1, :], clicks[1][:, :random_index+1])
                    else:
                        raise ValueError("Invalid click mode! Please enter single or multi")
                    optimizer.zero_grad()
                    image, gt2D, click, boxes = image.to(device), gt2D.to(device), (click[0].to(device), click[1].to(device)), boxes.to(device)
                    
                    if args.prompt_mode == "click_re":
                        logits_pred = None
                        for i in range(args.num_reselect):
                            if i != args.num_reselect - 1:
                                with torch.no_grad():
                                    logits_pred_tmp, _ = medsam_lite_model(image, None, click, None)
                                    # click = reselect_click(batch_size, clicks, logits_pred_tmp, click)
                            else:
                                del logits_pred_tmp
                                logits_pred, iou_pred = medsam_lite_model(image, None, click, None)
                                del click
                    elif args.prompt_mode == "click_mask":  
                        logits_pred0, _ = medsam_lite_model(image, None, click, None)
                        logits_pred1, _ = medsam_lite_model(image, None, click, logits_pred0)
                        logits_pred, iou_pred = medsam_lite_model(image, None, click, logits_pred1)
                        if is_main_host:
                            visualize_gt(image[0], gt2D[0], click, step, save_path=vis_val_epoch_dir)
                            visualize_pred(image[0], logits_pred0, click, step, 1, save_path=vis_val_epoch_dir)
                            visualize_pred(image[0], logits_pred1, click, step, 2, save_path=vis_val_epoch_dir)
                            visualize_pred(image[0], logits_pred, click, step, 3,  save_path=vis_val_epoch_dir)
                        del logits_pred0, logits_pred1
                    elif args.prompt_mode == "bbox":
                        logits_pred, iou_pred = medsam_lite_model(image, boxes, None, None)
                    else:
                        raise ValueError("Invalid prompt mode! Please enter click_mask or click_re or bbox")
                    val_loss = cal_loss(pred=(logits_pred, iou_pred),gt=gt2D, seg=(seg_loss, args.seg_loss_weight), ce=(ce_loss, args.ce_loss_weight), iou=(iou_loss, args.iou_loss_weight))
                    epoch_loss_val[step] = val_loss.item()
        epoch_end_time = time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_loss_world = [None for _ in range(world_size)]
        epoch_loss_world_val = [None for _ in range(world_size)]
        dist.all_gather_object(epoch_loss_world, epoch_loss)
        dist.all_gather_object(epoch_loss_world_val, epoch_loss_val)
        epoch_loss_reduced = np.vstack(epoch_loss_world).mean()
        epoch_loss_reduced_val = np.vstack(epoch_loss_world_val).mean()
        train_losses.append(epoch_loss_reduced)
        val_losses.append(epoch_loss_reduced_val)
        lr_scheduler.step(epoch_loss_reduced)
        if is_main_host and wandb_enable:
            wandb.log({"train_loss": epoch_loss_reduced,
                       "val_loss": epoch_loss_reduced_val})

        if is_main_host:
            module_revert_sync_BN = revert_sync_batchnorm(deepcopy(medsam_lite_model.module))
            weights = module_revert_sync_BN.state_dict()
            checkpoint = {
                "model": weights,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss_reduced,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_lite_latest.pth"))
        if epoch_loss_reduced < best_loss:
            if is_main_host:
                print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            if is_main_host:
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, join(model_save_path, "medsam_lite_best.pth"))
        dist.barrier()
        epoch_loss_reduced = 1e10
        # %% plot loss
        if is_main_host:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            axes[0].title.set_text("Dice + Binary Cross Entropy + IoU Loss")
            axes[0].plot(train_losses)
            axes[0].set_ylabel("Loss")
            axes[1].plot(epoch_times)
            axes[1].title.set_text("Epoch Duration")
            axes[1].set_ylabel("Duration (s)")
            axes[1].set_xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(join(model_save_path, "log.png"))
            plt.close()
        dist.barrier()
    if is_main_host and wandb_enable:
        wandb.finish(quiet=True)
        
# %%
if __name__ == "__main__":
    args = get_args()
    # sanity_check_dataset(args)
    main(args)
