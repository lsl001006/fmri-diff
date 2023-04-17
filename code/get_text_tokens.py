import argparse
from config import Config_MBM_finetune,Config_Generative_Model
from dataset_new import create_Kamitani_dataset, create_BOLD5000_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
from einops import rearrange
from torchvision.utils import save_image

def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning on Test fMRI', add_help=False)

    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mask_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--include_nonavg_test', type=bool)   
                        
    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

args = get_args_parser()
args = args.parse_args()
config = Config_Generative_Model()
config = update_config(args, config)

# -------------- transform functions ------------------
def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

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
# -------------------------------------------------
if config.dataset == 'GOD':
    fmri_latents_dataset_train, fmri_latents_dataset_test = create_Kamitani_dataset('../data/Kamitani/npz', 
                                                                                    config.roi, 
                                                                                    config.patch_size, 
                                                                                    fmri_transform=fmri_transform, 
                                                                                    image_transform=[img_transform_train, 
                                                                                                    img_transform_test], 
                                                                                    subjects=config.kam_subs)
    
elif config.dataset == 'BOLD5000':
    fmri_latents_dataset_train, fmri_latents_dataset_test = create_BOLD5000_dataset('../data/BOLD5000', 
                                                                                    config.patch_size, 
                                                                                    fmri_transform=fmri_transform, 
                                                                                    image_transform=[img_transform_train, 
                                                                                                    img_transform_test], 
                                                                                    subjects=config.bold5000_subs)

train_loader = DataLoader(fmri_latents_dataset_train, batch_size=2, shuffle=True)
test_loader = DataLoader(fmri_latents_dataset_test, batch_size=2, shuffle=False)  
for i, _dict in enumerate(train_loader):
    img = _dict['image'][0]
    img = unpreprocess(img)
    img = rearrange(img, 'h w c -> c h w')
    import pdb;pdb.set_trace()
    save_image(img, 'img.jpg')
    print(_dict['text'][0])                         
    
    


