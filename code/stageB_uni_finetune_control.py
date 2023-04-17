import os, sys
import numpy as np
import torch
import argparse
# from ml_collections import ConfigDict, ConfigFlags
import datetime
from absl import flags
from ml_collections import config_flags
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from uni_md.autoencoder import get_model
from uni_md.caption_decoder import CaptionDecoder
# from uni_md.unid_for_fmri_lowimg import uni_model, uniDM
from uni_md.unid_fmri_control import uniDM
from uni_md.clip import FrozenCLIPEmbedder
import copy
import clip

from config import Config_Generative_Model
# from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from dataset_new import create_Kamitani_dataset, create_BOLD5000_dataset


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img


def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img
    

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def main(config):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(config.checkpoint_path, exist_ok=True)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        random_crop(config.img_size-crop_pix, p=0.5),
        transforms.Resize((512, 512)), 
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)), 
        channel_last
    ])

    if config.dataset == 'GOD':
        fmri_latents_dataset_train, fmri_latents_dataset_test = create_Kamitani_dataset(config.kam_path, config.roi, config.patch_size, 
                fmri_transform=fmri_transform, image_transform=[img_transform_train, img_transform_test], 
                subjects=config.kam_subs)
        num_voxels = fmri_latents_dataset_train.num_voxels
    elif config.dataset == 'BOLD5000':
        fmri_latents_dataset_train, fmri_latents_dataset_test = create_BOLD5000_dataset(config.bold5000_path, config.patch_size, 
                fmri_transform=fmri_transform, image_transform=[img_transform_train, img_transform_test], 
                subjects=config.bold5000_subs)
        num_voxels = fmri_latents_dataset_train.num_voxels
    else:
        raise NotImplementedError

    # clip_text_model = FrozenCLIPEmbedder(device=device).to(device)
    # clip_text_model.eval()

    # clip_img_model, clip_img_model_preprocess = clip.load("/data/zbh/ckpt/clip/ViT-B-32.pt", device=device, jit=False)
    
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    autoencoder = get_model(**config.autoencoder).to(device)

    # prepare pretrained mbm 
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
    train_DM = uniDM(config, pretrain_mbm_metafile, num_voxels, device)
    train_DM.finetune(autoencoder, caption_decoder, fmri_latents_dataset_train, fmri_latents_dataset_test, config.batch_size)


def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--kam_path', type=str)
    parser.add_argument('--bold5000_path', type=str)
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # user setting
    parser.add_argument('--player_name', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)
    parser.add_argument('--checkpoint_path', type=str)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser


def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def update_collection(collection, config):
    for attr in collection:
        setattr(config, attr, getattr(collection, attr))
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=0):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger, 
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
            check_val_every_n_epoch=check_val_every_n_epoch)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.player_name == 'zbh':
        from configs.sample_unidiffuser_v1_zbh import get_config
    if args.player_name == 'lsl':
        from configs.sample_unidiffuser_v1_lsl import get_config
        
    config = Config_Generative_Model()

    uniconfig = get_config()
    config = update_config(args, config)
    config = update_collection(uniconfig, config)

    main(config)
