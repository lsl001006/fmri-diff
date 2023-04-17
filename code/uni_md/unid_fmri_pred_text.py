import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import itertools
import einops
import os
from torch.utils.data import DataLoader

from sc_mbm.mae_for_fmri import fmri_encoder
from uni_md.diffusion.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from uni_md.diffusion.uvit_fmri_text import fmri2text_nnet


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


def get_nnet(name, **kwargs):
    from uni_md.diffusion.uvit_multi_post_ln_v1 import UViT
    return UViT(**kwargs)


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def LSimple(x0, text_model, schedule, fmri):
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    eps_pred = text_model(xn, n, fmri)
    return mos(eps - eps_pred)


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


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


class fmri2text(nn.Module):
    def __init__(self, fmri_dim, text_length, text_dim):
        super().__init__()
        # for general, the fmri dim is 1280, text length is 77, and the text_dim is 64:
        inner_mlp_dim = 1536
        self.mlp = nn.Sequential(nn.Linear(fmri_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
                                 nn.SiLU(),
                                 nn.Linear(inner_mlp_dim, inner_mlp_dim),
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
                                  nn.SiLU(),
                                  nn.Conv1d(text_length, text_length, kernel_size=3, padding=1),
                                  nn.SiLU(),
                                  nn.Conv1d(text_length, text_length, kernel_size=3, padding=1),
                                  nn.SiLU())
        # add model for clip
    
    def forward(self, fmri):
        # fmri is (bs, 1, 1280)
        fmri_inner = self.mlp(fmri)
        fmri_inner = fmri_inner.view(fmri_inner.size(0), 64, -1)
        return self.conv(fmri_inner)


def obtain_text_from_image(img_encoder, clip_img_model, img_tensor, nnet, config, device, beta_):
    # beta_ = stable_diffusion_beta_schedule()
    # clip_img_model, clip_img_model_preprocess = clip.load("/data1/zbh/.cache/clip/ViT-B-32.pt", device=device, jit=False)
    
    # obtain the features from the input image
    ldm_z = img_encoder.encode(img_tensor)
    img_tensor_224 = F.interpolate(img_tensor, size=(224, 224))
    clip_img_feature = clip_img_model.encode_image(img_tensor_224)
    clip_img_feature = clip_img_feature.float().unsqueeze(1)

    # define the init text feature
    _text_init = torch.randn(img_tensor.size(0), 77, config.text_dim, device=device)

    # define the noise scheduler
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(beta_, device=device).float())

    # define diffusion process function
    def model_fmri_fn(x, timesteps, img_z, img_clip):
        text = x
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        z_out, clip_img_out, text_out = nnet(img_z, img_clip, text, t_img=t_img, t_text=timesteps,
                                            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        return text_out
    
    def model_fn(x, t_continuous):
        t = t_continuous * len(beta_)
        return model_fmri_fn(x, t, ldm_z, clip_img_feature)
    
    # define the dpm solver
    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

    # generation
    _x_init = _text_init
    print('generating text start!')
    with torch.no_grad():
        text = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / len(beta_), T=1.)
    return text


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


def obtain_text_from_fmri(text_model, fmri, config, device, beta_):
    text_model.eval()

    # define the init text feature
    _text_init = torch.randn(fmri.size(0), 77, config.text_dim, device=device)

    # define the noise scheduler
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(beta_, device=device).float())

    # define diffusion process function
    def model_fmri_fn(x, timesteps, fmri):
        text = x
        # t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        text_out = text_model(text, timesteps, fmri)
        return text_out
    
    def model_fn(x, t_continuous):
        t = t_continuous * len(beta_)
        return model_fmri_fn(x, t, fmri)
    
    # define the dpm solver
    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

    # generation
    _x_init = _text_init
    print('generating text start!')
    with torch.no_grad():
        text = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / len(beta_), T=1.)
    return text


class uniDM:
    def __init__(self, config, pretrain_mbm_metafile, num_voxels, device, epoch=500):
        # self.text_model = fmri2text(fmri_dim=1280, text_length=77, text_dim=config.text_dim).to(device)
        self.text_model = fmri2text_nnet().to(device)
        self.nnet = get_nnet(**config.nnet).to(device)
        self.beta = stable_diffusion_beta_schedule()
        self.schedule = Schedule(self.beta)
        self.nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'), strict=False)
        self.text_model.cond_stage_model = cond_stage_model(pretrain_mbm_metafile, num_voxels).to(device)
        self.config = config
        self.device = device
        self.epoch = epoch
        self.nnet.eval()
        # self.cond_stage_model.eval()
        self.optimizer = torch.optim.Adam(itertools.chain(self.text_model.parameters()), lr=2e-5, betas=(0.5, 0.999), weight_decay=0.0001)
    
    def finetune(self, img_encoder, caption_decoder, clip_img_model, dataset, test_dataset, bs1):
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        if os.path.isfile(os.path.join(self.config.checkpoint_path, 'fmri2text.pth')): #!= None:
            self.text_model.load_state_dict(torch.load(os.path.join(self.config.checkpoint_path, 'fmri2text.pth'), map_location='cpu'), strict=False)
            print('load fmri to text model successful!')
        
        for e in range(self.epoch):
            step = 0
            for _dict in dataloader:
                self.text_model.train()
                x = _dict['image'].to(self.device)
                fmri = _dict['fmri'].to(self.device)
                pred_gt_text = _dict['token'].to(self.device)
                # fmri = self.cond_stage_model(fmri) # b*1*1280
                x = x.permute(0,3,1,2).float()
                # x.requires_grad = False
                # with torch.no_grad():
                #     pred_gt_text = obtain_text_from_image(img_encoder, clip_img_model, x, self.nnet, self.config, self.device, self.beta)

                # training
                self.optimizer.zero_grad()
                # pred_fmri_text = self.text_model(fmri)
                # prompts_gt = caption_decoder.generate_captions(pred_gt_text)
                # prompts_ = caption_decoder.generate_captions(pred_fmri_text)
                # loss = torch.abs(pred_fmri_text - pred_gt_text).mean()
                loss = LSimple(x0=pred_gt_text, text_model=self.text_model, schedule=self.schedule, fmri=fmri).sum()

                if step % 20 == 0:
                    test_iter = iter(test_loader)
                    test_dict = next(test_iter)
                    test_x = test_dict['image'].to(self.device)
                    test_fmri = test_dict['fmri'].to(self.device)
                    test_x = test_x.permute(0,3,1,2).float()
                    # test_fmri_ = self.cond_stage_model(test_fmri[0:1, ...]) # b*1*1280
                    test_fmri_ = test_fmri[0:1, ...]
                    # pred_test_fmri = self.text_model(test_fmri_)
                    pred_test_fmri = obtain_text_from_fmri(self.text_model, test_fmri_, self.config, self.device, self.beta)
                    pred_z = obtain_img_from_text(pred_test_fmri, self.nnet, self.config, self.device, self.beta)
                    pred_img = img_encoder.decode(pred_z)
                    img = torch.cat((unpreprocess(test_x[0:1, ...]), unpreprocess(pred_img[0:1, ...])), dim=0)
                    prompts = caption_decoder.generate_captions(pred_test_fmri)
                    save_image(img, os.path.join(self.config.checkpoint_path, "%08d_.png" % step), nrow=1, normalize=False,
                           pad_value=0, range=(-1, 1))
                    with open(os.path.join(self.config.checkpoint_path, '%08d_prompts.txt' % step), 'w') as f:
                        print('\n'.join(prompts), file=f)

                print(f"epoch {e} step {step} loss is {loss.item()}")

                loss.backward()
                self.optimizer.step()
                step += 1

                if step % 20 == 0:
                    torch.save(self.text_model.state_dict(), os.path.join(self.config.checkpoint_path, 'fmri2text.pth'))
            torch.save(self.text_model.state_dict(), os.path.join(self.config.checkpoint_path, 'fmri2text.pth'))
