import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import einops
import os
import argparse
import torchvision.transforms as transforms
from sc_mbm.mae_for_fmri import fmri_encoder
import copy
import clip
from torchvision.utils import save_image
from einops import rearrange

from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from dc_ldm.models.diffusion.plms import PLMSSampler
from dc_ldm.ldm_for_fmri import fLDM
from uni_md.diffusion.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from uni_md.autoencoder import get_model
from uni_md.caption_decoder import CaptionDecoder
from uni_md.clip import FrozenCLIPEmbedder


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model


class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels, cond_dim=1280, global_pool=True):
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out


def get_nnet(name, **kwargs):
        from uni_md.diffusion.uvit_multi_post_ln_v1 import UViT
        return UViT(**kwargs)


def generation(img_encoder, caption_decoder, generative_model, dict, nnet, config, device, LDM_image):
    condition = generative_model.model.get_learned_conditioning(dict['fmri'][0:1, ...].to(device))
    beta_ = stable_diffusion_beta_schedule()
    # clip_text_model = FrozenCLIPEmbedder(device=device)
    # clip_text_model.eval()
    # clip_text_model.to(device)
    clip_img_model, clip_img_model_preprocess = clip.load("/data1/zbh/.cache/clip/ViT-B-32.pt", device=device, jit=False)
    ldm_z = img_encoder.encode(LDM_image)
    # LDM_image_224 = F.interpolate(LDM_image, size=(224, 224))
    LDM_image_224 = clip_img_model_preprocess(LDM_image)
    clip_img_feature = clip_img_model.encode_image(LDM_image_224)
    # LDM_clip_text = clip.decode()
    # import pdb; pdb.set_trace()
    clip_img_feature = clip_img_feature.float().unsqueeze(1)
    _z_init = torch.randn(1, *config.z_shape, device=device)
    _text_init = torch.randn(1, 77, config.text_dim, device=device)
    _clip_img_init = torch.randn(1, 1, config.clip_img_dim, device=device)

    # define the noise scheduler
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(beta_, device=device).float())

    def split_joint(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img, text = x.split([z_dim, config.clip_img_dim, 77 * config.text_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=config.text_dim)
        return z, clip_img, text

    def combine_joint(z, clip_img, text):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([z, clip_img, text], dim=-1)
    
    def split_1(x):
        clip_img, text = x.split([config.clip_img_dim, 77 * config.text_dim], dim=1)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=config.text_dim)
        return clip_img, text

    def combine_1(clip_img, text):
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([clip_img, text], dim=-1)
    
    def split_2(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img

    def combine_2(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)

    def model_fmri_fn_1(x, timesteps, condition, clip_condition):
        # clip, text = split_1(x)
        text = x
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        z_out, clip_img_out, text_out = nnet(condition, clip_condition, text, t_img=t_img, t_text=timesteps,
                                            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        # x_out = combine_1(clip_img_out, text_out)
        x_out = text_out
        # e_t = generative_model.model.apply_model(z[:, 0:3, ...], timesteps, condition)
        # ldm_w = timesteps/(len(beta_)*8)
        # z_out = (1 - ldm_w**2)**(0.5) * z_out + ldm_w * torch.cat((e_t, e_t.mean(dim=1, keepdim=True)), dim=1)
        
        return x_out
    
    def model_fmri_fn_2(x, timesteps, condition):
        z, clip = split_2(x)
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        z_out, clip_img_out, text_out = nnet(z, clip, condition, t_img=timesteps, t_text=t_img,
                                            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        x_out = combine_2(z_out, clip_img_out)
        
        return x_out

    # stage 1
    # define the sample function
    def model_fn_1(x, t_continuous):
        t = t_continuous * len(beta_)
        return model_fmri_fn_1(x, t, ldm_z, clip_img_feature)
    
    # define the dpm solver
    dpm_solver = DPM_Solver(model_fn_1, noise_schedule, predict_x0=True, thresholding=False)

    # generating
    # _x_init = combine_1(_clip_img_init, _text_init)
    _x_init = _text_init
    print('generating text start!')
    with torch.no_grad():
        x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / len(beta_), T=1.)
    text = x
    
    # stage 2
    # prompts = caption_decoder.generate_captions(text)
    # contexts = clip_text_model.encode(prompts)
    # text = caption_decoder.encode_prefix(contexts)
    def model_fn_2(x, t_continuous):
        t = t_continuous * len(beta_)
        return model_fmri_fn_2(x, t, text)
    
    dpm_solver = DPM_Solver(model_fn_2, noise_schedule, predict_x0=True, thresholding=False)
    x_init_2 = combine_2(_z_init, _clip_img_init)
    
    print('generating img start!')
    with torch.no_grad():
        x_2 = dpm_solver.sample(x_init_2, steps=config.sample.sample_steps, eps=1. / len(beta_), T=1.)
    z, _ = split_2(x_2)
    
    image = unpreprocess(img_encoder.decode(z))
    prompts = caption_decoder.generate_captions(text)
    return image, prompts


def LDM_generation(fmri, generative_model, ddim_steps, device):
    ldm_config = generative_model.ldm_config
    shape = (ldm_config.model.params.channels, 
             ldm_config.model.params.image_size, ldm_config.model.params.image_size)
    model = generative_model.model.to(device)
    sampler = PLMSSampler(model)

    fmri = fmri[0:1, ...].to(device)
    c = model.get_learned_conditioning(fmri)
    samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                    conditioning=c,
                                    batch_size=1,
                                    shape=shape,
                                    verbose=False)
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    return x_samples_ddim


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


def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


def main(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

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
    
    test_loader = DataLoader(fmri_latents_dataset_test, batch_size=len(fmri_latents_dataset_test), shuffle=False)

    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    autoencoder = get_model(**config.autoencoder).to(device)

    # LDM
    model_path = os.path.join('/expand_data/zbh/ckpt/frmi_pretrains/BOLD5000', 'finetuned.pth')
    sd = torch.load(model_path, map_location='cpu')
    config_sd = sd['config']
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
    generative_model = fLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config_sd.logger, 
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond)
    generative_model.model.to(device)
    generative_model.model.load_state_dict(sd['model_state_dict'])    # model is LDM

    # unidiffuser
    nnet = get_nnet(**config.nnet).to(device)
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), strict=False)
    nnet.eval()

    test_iter = iter(test_loader)
    test_dict = next(test_iter)
    
    image = LDM_generation(test_dict['fmri'], generative_model, config.ddim_steps, device)
    image = F.interpolate(image, size=(512, 512))
    image_2, prompts = generation(img_encoder=autoencoder, caption_decoder=caption_decoder, generative_model=generative_model,
                              dict=test_dict, nnet=nnet, config=config, device=device, LDM_image=unpreprocess(image))
    # import pdb; pdb.set_trace()
    img = torch.cat((unpreprocess(image), image_2, unpreprocess(test_dict['image'][0:1, ...].permute(0, 3, 1, 2).to(device))), dim=0)
    save_image(img, os.path.join(config.checkpoint_path, "test_.png"), nrow=3, normalize=False,
            pad_value=0, range=(-1, 1))
    with open(os.path.join(config.checkpoint_path, 'test_prompts.txt'), 'w') as f:
        print('\n'.join(prompts), file=f)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    from configs.sample_unidiffuser_v1_zbh import get_config
    from config import Config_Generative_Model

    config = Config_Generative_Model()

    uniconfig = get_config()
    config = update_config(args, config)
    config = update_collection(uniconfig, config)

    main(config)
