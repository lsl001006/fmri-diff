import torch
import torch.nn as nn
import math
from ..timm_ import trunc_normal_, DropPath, Mlp
import einops
import torch.utils.checkpoint
import torch.nn.functional as F
# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
    print('xformers enabled')
except:
    XFORMERS_IS_AVAILBLE = False
    print('xformers disabled')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, in_chans):
    patch_size = int((x.shape[2] // in_chans) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


def interpolate_pos_emb(pos_emb, old_shape, new_shape):
    pos_emb = einops.rearrange(pos_emb, 'B (H W) C -> B C H W', H=old_shape[0], W=old_shape[1])
    pos_emb = F.interpolate(pos_emb, new_shape, mode='bilinear')
    pos_emb = einops.rearrange(pos_emb, 'B C H W -> B (H W) C')
    return pos_emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if XFORMERS_IS_AVAILBLE:  # the xformers lib allows less memory, faster training and inference
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim) if skip else None
        self.norm2 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
            x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))
        x = self.norm3(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class AlignConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, in_channel=1, out_channel=1024):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim),
                                 nn.SiLU(),
                                 nn.Linear(input_dim, input_dim),
                                 nn.SiLU(),
                                 nn.Linear(input_dim, output_dim),
                                 nn.SiLU()) # 1025*1024
        self.mid_channel = 512
        self.conv = nn.Sequential(nn.Conv1d(in_channel, self.mid_channel, kernel_size=3, padding=1),
                                  nn.SiLU(),
                                  nn.Conv1d(self.mid_channel, self.mid_channel, kernel_size=3, padding=1),
                                  nn.SiLU(),
                                  nn.Conv1d(self.mid_channel, out_channel, kernel_size=3, padding=1),
                                  nn.SiLU())
        self.merge_conv_ = nn.Sequential(nn.Conv1d(out_channel*2, out_channel, kernel_size=3, padding=1),
                                         nn.SiLU(),
                                         nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
                                         nn.SiLU())
    
    def forward(self, image, fmri_clip, contri_w = 0.9):
        fmri_deal = self.conv(self.mlp(fmri_clip))
        fmri_deal = torch.cat((image, fmri_deal), dim=1)
        return self.merge_conv_(fmri_deal)


class AlignNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.selection_layer = nn.Sequential(
                                            nn.Linear(input_dim, 128),
                                            nn.SiLU(),
                                            nn.Linear(128,256),
                                            nn.SiLU(),
                                            nn.Linear(256,512),
                                            nn.SiLU(),
                                            nn.Linear(512, output_dim),
                                            ) # 1025*1024
        self.norm = nn.LayerNorm(output_dim)
        # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, input_dim, input_length)
        output: tensor of shape (batch_size, output_dim, input_length)
        """
        mean_input, std_input = x.mean(), x.std()
        x = x.transpose(-1,-2) # b*1536*1025
        # First, use a linear layer to map the input tensor to a (batch_size, output_dim, input_length) tensor
        x = self.selection_layer(x) # b*1536*1024
        x = self.norm(x)
        x = x*std_input+mean_input
        output = x.transpose(-1,-2)
        return output


class SimpleLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b1 = nn.Linear(dim, dim)
        self.an = nn.SiLU()
    
    def forward(self, x_in, shortcut=None):
        if shortcut == None:
            return self.an(self.b1(x_in) + x_in)
        else:
            return self.an(self.b1(x_in) + shortcut)


class MultiLayerFMRI(nn.Module):
    def __init__(self, depth, in_dim=512, out_dim=1536):
        super().__init__()
        self.first_model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )
        self.in_blocks = nn.ModuleList(
            [SimpleLayer(out_dim) for _ in range(depth)]
        )
        self.out_blocks = nn.ModuleList(
            [SimpleLayer(out_dim) for _ in range(depth)]
        )
    
    def forward(self, fmri_c):
        c = self.first_model(fmri_c)
        in_features = []
        out_features = []
        for blk in self.in_blocks:
            c = blk(c)
            in_features.append(c)
        in_features_ = in_features[:]
        for blk in self.out_blocks:
            c = blk(c, in_features_.pop())
            out_features.append(c)
        return in_features, out_features


class UViT(nn.Module):
    def __init__(self, img_size, in_chans, patch_size, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, pos_drop_rate=0., drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, mlp_time_embed=False, use_checkpoint=False,
                 text_dim=None, num_text_tokens=None, clip_img_dim=None):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size  # the default img size
        assert self.img_size[0] % patch_size == 0 and self.img_size[1] % patch_size == 0
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.time_img_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.time_text_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.text_embed = nn.Linear(text_dim, embed_dim)
        self.text_out = nn.Linear(embed_dim, text_dim)

        self.clip_img_embed = nn.Linear(clip_img_dim, embed_dim)
        self.clip_img_out = nn.Linear(embed_dim, clip_img_dim)

        self.num_text_tokens = num_text_tokens
        self.num_tokens = 1 + 1 + num_text_tokens + 1 + self.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, skip=True, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.token_embedding = nn.Embedding(2, embed_dim)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.align_model = AlignNet(1025,1024) # cat
        # self.align_conv_model = AlignConvNet(input_dim=1024, output_dim=1536, in_channel=3, out_channel=1024) # addition
        # self.multi_layer_model = MultiLayerFMRI(depth=depth//2, in_dim=1024)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, img, clip_img, text, low_img, t_img, t_text, data_type):
        
        _, _, H, W = img.shape # b*4*64*64

        img = self.patch_embed(img) # bs, 1024, 1536
        # low_img = low_img.view(low_img.size(0), low_img.size(1), -1)
        # img = self.align_conv_model(img, low_img)

        t_img_token = self.time_img_embed(timestep_embedding(t_img, self.embed_dim))
        t_img_token = t_img_token.unsqueeze(dim=1)
        t_text_token = self.time_text_embed(timestep_embedding(t_text, self.embed_dim))
        t_text_token = t_text_token.unsqueeze(dim=1)

        text = self.text_embed(text)
        
        ori_clip_img = clip_img.clone() # b*1*512
        clip_img = self.clip_img_embed(clip_img)

        token_embed = self.token_embedding(data_type).unsqueeze(dim=1)

        x = torch.cat((t_img_token, t_text_token, token_embed, text, clip_img, img), dim=1)

        num_text_tokens, num_img_tokens = text.size(1), img.size(1)

        pos_embed = torch.cat(
            [self.pos_embed[:, :1 + 1, :], self.pos_embed_token, self.pos_embed[:, 1 + 1:, :]], dim=1)
        if H == self.img_size[0] and W == self.img_size[1]:
            pass
        else:  # interpolate the positional embedding when the input image is not of the default shape
            pos_embed_others, pos_embed_patches = torch.split(pos_embed, [1 + 1 + 1 + num_text_tokens + 1, self.num_patches], dim=1)
            pos_embed_patches = interpolate_pos_emb(pos_embed_patches, (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size),
                                                    (H // self.patch_size, W // self.patch_size))
            pos_embed = torch.cat((pos_embed_others, pos_embed_patches), dim=1)

        # multi layer condition
        # in_features, out_features = self.multi_layer_model(ori_clip_img)
        # low_img = low_img.view(low_img.size(0), low_img.size(1), -1)
        # in_features, out_features = self.multi_layer_model(low_img)

        x = x + pos_embed
        x = self.pos_drop(x)

        skips = []
        for i, blk in enumerate(self.in_blocks):
            # x = 0.8 * blk(x) + 10 * (in_features[i].mean(dim=1, keepdim=True))
            x = blk(x)
            # import pdb; pdb.set_trace()
            skips.append(x)

        x = self.mid_block(x)

        for i, blk in enumerate(self.out_blocks):
            # x = 0.8 * blk(x, skips.pop()) + 10 * (out_features[i].mean(dim=1, keepdim=True))
            x = blk(x, skips.pop())
            # import pdb; pdb.set_trace()

        x = self.norm(x)
        
        # after split: text_out:b*77*1536, clip_img_out:b*1*1536, img_out:b*1024*1536
        t_img_token_out, t_text_token_out, token_embed_out, text_out, clip_img_out, img_out = x.split((1, 1, 1, num_text_tokens, 1, num_img_tokens), dim=1)

        # here add the model to deal with the image and clip noise @lsl
        # network input : img_out:b*1024*1536, ori_clip_img(without embedded):b*1*512
        # network output: img_out:b*1024*1536
        
        # ##### cat
        # ori_clip_img -= ori_clip_img.mean() # mean to 0
        # ori_clip_img /= ori_clip_img.std()
        # ori_clip_img = img_out.std()*ori_clip_img+img_out.mean() # 和img_out取值范围相同
        # fusion_img = torch.cat([img_out, ori_clip_img.repeat(1,1,3)], dim=1)
        # img_out_new = self.align_model(fusion_img)

        ##### add
        # low_img = low_img.view(low_img.size(0), low_img.size(1), -1)
        # fmri_cond = self.align_conv_model(img_out, low_img)
        # img_out = 0.6*img_out + 8*fmri_cond

        img_out = self.decoder_pred(img_out)
        img_out = unpatchify(img_out, self.in_chans)

        clip_img_out = self.clip_img_out(clip_img_out)

        text_out = self.text_out(text_out)
        return img_out, clip_img_out, text_out
