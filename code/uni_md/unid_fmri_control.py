import torch
import torch.nn as nn
import numpy as np
import einops
import itertools
from torch.utils.data import DataLoader
import os
import clip

from dc_ldm.ldm_for_fmri_clip import cond_stage_model
from uni_md.diffusion.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from PIL import Image
from torchvision.utils import save_image


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
    

def Uni_LSimple(img_0, clip_, text, fmri, nnet, schedule, config, device, **kwargs):
    N = schedule.N

    n, eps, xn = schedule.sample(img_0)
    t_text = torch.zeros(n.size(0), dtype=torch.int, device=device)
    z_out, clip_img_out, text_out = nnet(xn, clip_, text=text, fmri=fmri, t_img=n, t_text=t_text,
                                         data_type=torch.zeros_like(n, device=device, dtype=torch.int) + config.data_type)

    return mos(eps - z_out).sum()


def get_nnet(name, **kwargs):
    from uni_md.diffusion.uvit_fmri_control_update0416 import UViT
    return UViT(**kwargs)


def obtain_img_from_text(text_tensor, nnet, config, device, beta_):
    nnet.eval()

    # define the init image feature
    _z_init = torch.randn(1, *config.z_shape, device=device)
    _clip_img_init = torch.randn(1, 1, config.clip_img_dim, device=device)

    # define the noise scheduler
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(beta_, device=device).float())

    def split(x):
        # split the clip image and the latent image
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img
    
    def combine(z, clip_img):
        # combine the clip_image and the latent image
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)

    # define diffusion process function
    def model_fmri_fn(x, timesteps, text):
        z, clip_img = split(x)
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        z_out, clip_img_out, text_out = nnet(z, clip_img, text, t_img=timesteps, t_text=t_text,
                                            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        out_ = combine(z_out, clip_img_out)
        return out_
    
    def model_fn(x, t_continuous):
        t = t_continuous * len(beta_)
        return model_fmri_fn(x, t, text_tensor)
    
    # define the dpm solver
    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

    # generation
    _x_init = combine(_z_init, _clip_img_init)
    print('generating text start!')
    with torch.no_grad():
        x_out = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / len(beta_), T=1.)
    z_out, _ = split(x_out)
    return z_out



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
        
        self.copy_weights()

    def copy_weights(self):
        scratch_dict = self.nnet.controlnet.state_dict()
        target_dict = {}
        for k in scratch_dict.keys():
            if k in self.nnet.state_dict().keys():
                target_dict[k] = self.nnet.state_dict()[k].clone()
            else:
                target_dict[k] = scratch_dict[k].clone()
                print(f'These weights are newly added: {k}')
        self.nnet.controlnet.load_state_dict(target_dict, strict=True)
                
    
    def forward(self, z_img, clip_, text, fmri, device):
        fmri = self.cond_stage_model(fmri) # b*1*1280
        loss = Uni_LSimple(z_img, clip_, text, fmri, self.nnet, self.schedule, self.config, device)
        return loss


class uniDM:
    def __init__(self, config, pretrain_mbm_metafile, num_voxels, device, epoch=100):
        self.model = uni_model(config).to(device)
        self.config = config
        self.epoch = epoch
        self.device = device
        self.model.cond_stage_model = cond_stage_model(pretrain_mbm_metafile, num_voxels).to(device)
        # self.model.cond_stage_model.eval() # ?
        self.model.nnet.eval()
        if config.player_name == 'lsl':
            self.clip_image_model, self.clip_preprocess = clip.load("/mnt/public/usr/lishanglin/Research/fmri/models/ViT-B-32.pt", device=device, jit=False)
        elif config.player_name == 'zbh':
            self.clip_image_model, self.clip_preprocess = clip.load("/data/zbh/ckpt/clip/ViT-B-32.pt", device=device, jit=False)
        self.optimizer = torch.optim.Adam(itertools.chain(self.model.nnet.controlnet.parameters(),
                                                          self.model.cond_stage_model.parameters()), 
                                                          lr=2e-5, betas=(0.5, 0.999), weight_decay=0.0001) #

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
    
    def model_fmri_fn(self, x, timesteps, text, fmri):
        z, clip_ = self.split(x)
        self.model.nnet.eval()
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)
        z_out, clip_img_out, text_out = self.model.nnet(z, clip_, text=text, fmri=fmri, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(timesteps, device=self.device, dtype=torch.int) + self.config.data_type)
        return self.combine(z_out, clip_img_out)

    def generated_image(self, img_encoder, caption_decoder, text_embed, fmri):
        _z_init = torch.randn(1, *self.config.z_shape, device=self.device)
        _clip_img_init = torch.randn(1, 1, self.config.clip_img_dim, device=self.device)
        _x_init = self.combine(_z_init, _clip_img_init)
        fmri = self.model.cond_stage_model(fmri)
        clip_ = torch.randn(1, 1, self.config.clip_img_dim, device=self.device)

        # define the noise scheduler
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self.model.beta, device=self.device).float())

        # define the sample function
        def model_fn(x, t_continuous):
            t = t_continuous * len(self.model.beta)
            return self.model_fmri_fn(x, t, text_embed, fmri)

        # define the dpm solver
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

        # generating
        print('generating start!')
        with torch.no_grad():
            # with torch.autocast(device_type=self.device):
                x = dpm_solver.sample(_x_init, steps=self.config.sample.sample_steps, eps=1. / len(self.model.beta), T=1.)
        z, _ = self.split(x)
        image = unpreprocess(img_encoder.decode(z))
        prompts = caption_decoder.generate_captions(text_embed)
        return image, prompts

    
    def finetune(self, img_encoder, caption_decoder, dataset, test_dataset, bs1):
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        if os.path.isfile(os.path.join(self.config.checkpoint_path, 'uniDM_fri.pth')):
            self.model.load_state_dict(torch.load(os.path.join(self.config.checkpoint_path, 'uniDM_fri.pth'), map_location='cpu'), strict=False)
            print('load fmri model successful!')
        else:
            print('train from scatch...')

        for e in range(self.epoch):
            step = 0
            for _dict in dataloader:
                self.model.nnet.train()
                # gt
                x = _dict['image'].to(self.device)
                fmri = _dict['fmri'].to(self.device)
                text_token = _dict['token'].to(self.device)
                
                clip_images = []
                for b in range(x.shape[0]):
                    x_pil = Image.fromarray(255*(unpreprocess(x[b].float()).cpu().numpy()).astype(np.uint8))
                    clip_image = self.clip_image_model.encode_image(self.clip_preprocess(x_pil).unsqueeze(0).to(self.device))
                    clip_images.append(clip_image)
                clip_images = torch.stack(clip_images)

                x = x.permute(0,3,1,2).float() # b,h,w,c -> b,c,h,w
                z_img = img_encoder.encode(x)
                
                self.optimizer.zero_grad()

                # text_noise = torch.randn(x.size(0), 77, self.config.text_dim, device=self.device)
                loss = self.model(z_img, clip_images, text_token, fmri, self.device)
                loss = loss.sum()

                if step % 100 == 0 or loss < 0.3:
                    test_iter = iter(test_loader)
                    test_dict = next(test_iter)
                    test_x = test_dict['image'].to(self.device)
                    test_fmri = test_dict['fmri'].to(self.device)
                    test_token = test_dict['token'].to(self.device)
                    test_x = test_x.permute(0,3,1,2).float()
                    image, prompts = self.generated_image(img_encoder, caption_decoder, test_token[0:1, ...], test_fmri[0:1, ...])
                    img = torch.cat((image, unpreprocess(test_x[0:1, ...])), dim=0)
                    save_image(img, os.path.join(self.config.checkpoint_path, f"{e}_{step}_.png"), nrow=2, normalize=False, pad_value=0, range=(-1, 1))
                    with open(os.path.join(self.config.checkpoint_path, f"{e}_{step}_prompts.txt"), 'w') as f:
                        print('\n'.join(prompts), file=f)

                if step % 10 == 0:
                    print(f"epoch {e} step {step} loss is {loss.item()}")

                loss.backward()
                self.optimizer.step()
                step += 1
            torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_path, f'uniDM_fri.pth'))

