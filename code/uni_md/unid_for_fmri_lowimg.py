import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import einops
import pytorch_lightning as pl
from sc_mbm.mae_for_fmri import fmri_encoder
from dc_ldm.ldm_for_fmri import fLDM
from uni_md.diffusion.uvit_multi_post_ln_v1 import UViT
from uni_md.diffusion.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import os
import itertools


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


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


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


def LSimple(x0, nnet, schedule, **kwargs):
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    eps_pred = nnet(xn, n, **kwargs)
    return mos(eps - eps_pred)


def Uni_LSimple(img_0, fmri_clip, fmri_text, low_img, nnet, schedule, config, device, **kwargs):
    N = schedule.N
    # n, eps, xn = schedule.sample(img_0)
    n, eps, xn = schedule.sample(fmri_text)
    t_img = torch.zeros(n.size(0), dtype=torch.int, device=device)
    clip_ = torch.randn(img_0.size(0), 1, config.clip_img_dim, device=device)
    # z_out, clip_img_out, text_out = nnet(xn, fmri_clip, text=text_noise, low_img=low_img, t_img=n, t_text=torch.ones_like(n) * N,
    #                                      data_type=torch.zeros_like(n, device=device, dtype=torch.int) + config.data_type)
    z_out, clip_img_out, text_out = nnet(img_0, clip_, text=xn, low_img=low_img, t_img=t_img, t_text=n,
                                         data_type=torch.zeros_like(n, device=device, dtype=torch.int) + config.data_type)

    # add quick distillation
    # xn_patch = nnet.patch_embed(xn)
    # low_img = low_img.view(low_img.size(0), low_img.size(1), -1)
    # xn_patch_pred = nnet.align_conv_model(xn_patch, low_img)

    return mos(eps - text_out).sum() # + torch.abs(xn_patch - xn_patch_pred).mean()


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
    if name == 'uvit_multi_post_ln_v1':
        from uni_md.diffusion.uvit_multi_post_ln_v1 import UViT
        return UViT(**kwargs)
    elif name == 'uvit_fmri':
        # from uni_md.diffusion.uvit_fmri import UViT
        from uni_md.diffusion.uvit_fmri_zbh import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


class fmri2text(nn.Module):
    def __init__(self, fmri_dim, text_length, text_dim):
        super().__init__()
        # for general, the fmri dim is 1280, text length is 77, and the text_dim is 64:
        inner_mlp_dim = 512
        self.mlp = nn.Sequential(nn.Linear(fmri_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, text_dim*64),
                                 nn.SiLU())
        self.conv = nn.Sequential(nn.Conv1d(64, text_length, kernel_size=3, padding=1),
                                  nn.SiLU(),
                                  nn.Conv1d(text_length, text_length, kernel_size=3, padding=1),
                                  nn.SiLU(),
                                  nn.Conv1d(text_length, text_length, kernel_size=3, padding=1),
                                  nn.SiLU())
    
    def forward(self, fmri):
        # fmri is (bs, 1, 1280)
        fmri_inner = self.mlp(fmri)
        fmri_inner = fmri_inner.view(fmri_inner.size(0), 64, -1)
        return self.conv(fmri_inner)


class uni_model(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.beta = stable_diffusion_beta_schedule()
        self.schedule = Schedule(self.beta)
        self.cond_stage_model = None
        self.config = config
        self.nnet = get_nnet(**config.nnet)
        self.nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), strict=False)
        self.cond_deal = nn.Sequential(nn.Linear(1280, 512),
                                           nn.SiLU(),
                                           nn.Linear(512, 512),
                                           nn.SiLU(),
                                           nn.Linear(512, 512),
                                           nn.SiLU(),
                                           nn.Linear(512, 512),
                                           nn.SiLU(),
                                           nn.Linear(512, config.clip_img_dim),
                                           nn.SiLU())
        self.text_deal = fmri2text(fmri_dim=1280, text_length=77, text_dim=config.text_dim)
    
    def forward(self, z_img, fmri, text_noise, low_img, device):
        fmri = self.cond_stage_model(fmri) # b*1*1280
        # import pdb; pdb.set_trace()
        fmri_clip = self.cond_deal(fmri)
        fmri_text = self.text_deal(fmri)
        loss = Uni_LSimple(z_img, fmri_clip, fmri_text, low_img, self.nnet, self.schedule, self.config, device)
        return loss


class uniDM:
    def __init__(self, config, pretrain_mbm_metafile, num_voxels, device, epoch=100):
        self.model = uni_model(config).to(device)
        self.config = config
        self.epoch = epoch
        self.device = device
        self.model.cond_stage_model = cond_stage_model(pretrain_mbm_metafile, num_voxels).to(device)
        self.model.cond_stage_model.eval()
        # self.model.nnet.eval()
        # self.optimizer = torch.optim.Adam(itertools.chain(self.model.nnet.align_model.parameters(), self.model.cond_deal.parameters()), 
        #                                     lr=2e-5, betas=(0.5, 0.999), weight_decay=0.0001)
        # self.optimizer = torch.optim.Adam(itertools.chain(self.model.nnet.align_conv_model.parameters(), self.model.cond_deal.parameters()), 
        #                                     lr=2e-5, betas=(0.5, 0.999), weight_decay=0.0001)
        self.optimizer = torch.optim.Adam(itertools.chain(self.model.text_deal.parameters(), self.model.cond_deal.parameters()), 
                                            lr=2e-5, betas=(0.5, 0.999), weight_decay=0.0001)

    def split_joint(self, x):
        # split the latent image and the text
        C, H, W = self.config.z_shape
        z_dim = C * H * W
        # z, clip_img, text = x.split([z_dim, self.config.clip_img_dim, 77 * self.config.text_dim], dim=1)
        z, text = x.split([z_dim, 77 * self.config.text_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=self.config.text_dim)
        return z, text
    
    def combine_joint(self, z, text):
        # combine the latent image and the text
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([z, text], dim=-1)
    
    def split(self, x):
        # split the clip image and the latent image
        C, H, W = self.config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, self.config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=self.config.clip_img_dim)
        return z, clip_img
    
    def combine(self, z, clip_img):
        # combine the clip_image and the latent image
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)
    
    def model_fmri_fn(self, x, timesteps, fmri_clip, low_img, fmri_text):
        # z, text = self.split_joint(x)
        z, clip = self.split(x)
        self.model.nnet.eval()
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)
        z_out, clip_img_out, text_out = self.model.nnet(z, clip, text=fmri_text, low_img=low_img, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)
        return self.combine(z_out, clip_img_out)

    def generated_image(self, img_encoder, caption_decoder, fmri, low_img):
        _z_init = torch.randn(1, *self.config.z_shape, device=self.device)
        _text_init = torch.randn(1, 77, self.config.text_dim, device=self.device)
        _clip_img_init = torch.randn(1, 1, self.config.clip_img_dim, device=self.device)
        # _x_init = self.combine_joint(_z_init, _text_init)
        _x_init = self.combine(_z_init, _clip_img_init)
        fmri = self.model.cond_stage_model(fmri)
        fmri_clip = self.model.cond_deal(fmri)
        fmri_text = self.model.text_deal(fmri)

        # define the noise scheduler
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self.model.beta, device=self.device).float())

        # define the sample function
        def model_fn(x, t_continuous):
            t = t_continuous * len(self.model.beta)
            return self.model_fmri_fn(x, t, fmri_clip, low_img, fmri_text)

        # define the dpm solver
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

        # generating
        print('generating start!')
        with torch.no_grad():
            # with torch.autocast(device_type=self.device):
                x = dpm_solver.sample(_x_init, steps=self.config.sample.sample_steps, eps=1. / len(self.model.beta), T=1.)
        z, _ = self.split(x)
        image = unpreprocess(img_encoder.decode(z))
        # prompts = caption_decoder.generate_captions(text)
        prompts = caption_decoder.generate_captions(fmri_text)
        return image, prompts

    def finetune(self, img_encoder, caption_decoder, dataset, test_dataset, bs1):
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        if os.path.isfile(os.path.join(self.config.checkpoint_path, 'uniDM_fri.pth')): #!= None:
            self.model.load_state_dict(torch.load(os.path.join(self.config.checkpoint_path, 'uniDM_fri.pth'), map_location='cpu'), strict=False)
            print('load fmri model successful!')
        
        for e in range(self.epoch):
            step = 0
            for _dict in dataloader:
                self.model.nnet.train()
                x = _dict['image'].to(self.device)
                fmri = _dict['fmri'].to(self.device)
                x = x.permute(0,3,1,2).float()
                self.optimizer.zero_grad()
                z_img = img_encoder.encode(x)
                # clip_img_noise = torch.randn(x.size(0), 1, self.config.clip_img_dim, device=self.device)
                text_noise = torch.randn(x.size(0), 77, self.config.text_dim, device=self.device)
                # add image condition
                low_img = F.interpolate(x, size=(32, 32))
                
                loss = self.model(z_img, fmri, text_noise, low_img, self.device)
                loss = loss.sum()

                if step % 100 == 0:
                    test_iter = iter(test_loader)
                    test_dict = next(test_iter)
                    test_x = test_dict['image'].to(self.device)
                    test_fmri = test_dict['fmri'].to(self.device)
                    test_x = test_x.permute(0,3,1,2).float()
                    low_img_test = F.interpolate(test_x, size=(32, 32))
                    image, prompts = self.generated_image(img_encoder, caption_decoder, test_fmri[1:2, ...], low_img_test[1:2, ...])
                    
                    # view train dataset
                    train_image, train_prompts = self.generated_image(img_encoder, caption_decoder, fmri[0:1, ...], low_img[0:1, ...])

                    prompts.extend(train_prompts)
                    img = torch.cat((image, unpreprocess(test_x[1:2, ...]), train_image, unpreprocess(x[0:1, ...])), dim=0)
                    save_image(img, os.path.join(self.config.checkpoint_path, "%08d_.png" % step), nrow=2, normalize=False,
                           pad_value=0, range=(-1, 1))
                    with open(os.path.join(self.config.checkpoint_path, '%08d_prompts.txt' % step), 'w') as f:
                        print('\n'.join(prompts), file=f)

                if step % 10 == 0:
                    print(f"epoch {e} step {step} loss is {loss.item()}")

                loss.backward()
                self.optimizer.step()
                step += 1
            torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_path, 'uniDM_fri.pth'))
