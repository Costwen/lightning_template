# add 1x1 conv to resblock
from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from models.op.common import *
import xformers.ops as xops
from xformers.components.positional_embedding import RotaryEmbedding
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes extra things to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, **kwargs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                args = dict(emb=emb)
                args['emb'] = emb
            elif isinstance(layer, FactorizedTransformer):
                args = kwargs
            else:
                args = {}
            x = layer(x, **args)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(self,query_dim: int,
        heads: int,
        dim_head: int,
        T: int,
        context_dim = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.positional_embedding = nn.Parameter(th.randn(T, query_dim) / inner_dim ** 0.5)
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.T = T

    def forward(self, x, context=None, mask=None, pos_bias=None):
        # x: (b h w) t c
        T = self.T
        h = self.heads
        if context is not None:
            context = repeat(context, 'b n d -> (b t) n d', t=T)
        context = default(context, x)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))
        if pos_bias:
            q, k = pos_bias(q, k)
            q, k = q.to(v.dtype), k.to(v.dtype)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if mask is not None:
            mask = repeat(mask, 'i j -> b n i j', b = q.shape[0], n = h)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = rearrange(out, 'b n h d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., T=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.T = T
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, pos_bias=None):
        h = self.heads
        T = self.T
        if context is not None and context.shape[0] != x.shape[0]:
            context = repeat(context, 'b n d -> (b t) n d', t=T)
        context = default(context, x)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)        

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h).contiguous(), (q, k, v))
        out = xops.memory_efficient_attention(q, k, v)
        out = rearrange(out, 'b n h d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False, T=None, mode="spatial"):
        super().__init__()

        if mode == "spatial":
            self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, T=T)  # is a self-attention
            self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=dropout,T=T)  # is self-attn if context is none
        else:
            self.attn1 = AttentionPool2d(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, T=T)  # is a self-attention
            self.attn2 = AttentionPool2d(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,T=T)
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, pos_bias=None):
        x = self.attn1(self.norm1(x), mask=mask, pos_bias=pos_bias) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask, pos_bias=pos_bias) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, channels, num_head_channels,
                 depth=1, dropout=0., context_dim=None, checkpoint=False,T=None):
        super().__init__()
        self.channels = channels
        inner_dim = channels
        d_head = num_head_channels
        n_heads = inner_dim // d_head
        self.norm = Normalize(channels)
        self.T = T

        self.proj_in = nn.Linear(channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=checkpoint, T=T)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Linear(channels, inner_dim))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in

class TemporalTransformer(nn.Module):
    def __init__(self, channels, num_head_channels,
                 depth=1, dropout=0., context_dim=None, checkpoint=False,T=None, model_channels=None):
        super().__init__()
        self.channels = channels
        inner_dim = channels
        d_head = num_head_channels
        n_heads = inner_dim // d_head
        self.norm = Normalize(channels)
        self.T = T
        self.proj_in = nn.Linear(inner_dim, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=None, checkpoint=checkpoint, T=T, mode="temporal")
                for d in range(depth)]
        )
        # self.rel_bias = ContinuousPositionBias(dim = inner_dim // 2, heads = n_heads, num_dims = 1)
        self.pos_bias = RotaryEmbedding(d_head)
        self.proj_out = zero_module(nn.Linear(inner_dim, inner_dim))
        
    def forward(self, x, mask=None, context=None, is_im=False):
        bt, c, h, w = x.shape
        if is_im:
            eye_mask = torch.eye(self.T, dtype=x.dtype, device=x.device)
            min_value = torch.finfo(eye_mask.dtype).min
            mask = eye_mask.masked_fill(eye_mask == 0, min_value)
        x = rearrange(x, '(b t) c h w -> (b h w) c t', t=self.T)
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.proj_in(x) # just a linear projection
        for block in self.transformer_blocks:
            x = block(x, mask=mask, pos_bias=self.pos_bias)
        x = self.proj_out(x)
        x = rearrange(x, 'b t c -> b c t')
        out = x + x_in
        out = rearrange(out, '(b h w) c t -> (b t) c h w', h=h, w=w)
        return out

class Conv1D(nn.Module):
    def __init__(self, T, channels, kernel_size=3, dropout=0.):
        super().__init__()
        self.T = T
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
    def forward(self, x):
        bt, c, h, w = x.shape
        x = rearrange(x, '(b t) c h w -> (b h w) c t', t=self.T)
        x = self.conv(x) + x
        x = rearrange(x, '(b h w) c t -> (b t) c h w', h=h, w=w)   
        return x
        
class Pseudo3DConv(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        T
    ):
        super().__init__()
        self.T = T
        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size, padding=kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)
    def forward(
        self,
        x,
        is_im = False
    ):
        bt, c, h, w = x.shape
        x = self.spatial_conv(x)
        x = rearrange(x, '(b t) c h w -> (b h w) c t', t=self.T)
        x = self.temporal_conv(x)
        x = rearrange(x, '(b h w) c t -> (b t) c h w', h=h, w=w) 
        return x

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 1,
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, *dimensions):
        device = self.device

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        time_embed_dim,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        T = None,
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.T = T

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
            # zero_module(Conv1D(T, self.out_channels, kernel_size=3, dropout=dropout)),
            Pseudo3DConv(channels, self.out_channels, 3, T),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                time_embed_dim,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            # zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
            # zero_module(Conv1D(T, self.out_channels, kernel_size=3, dropout=dropout))
            Pseudo3DConv(self.out_channels, self.out_channels, 3, T),
            )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
            
        elif use_conv:
            # self.skip_connection = conv_nd(
            #     dims, channels, self.out_channels, 3, padding=1
            # )
            self.skip_connection = Pseudo3DConv(channels, self.out_channels, 3, T)
        else:
            # self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.skip_connection = Pseudo3DConv(channels, self.out_channels, 1, T)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class FactorizedTransformer(nn.Module):

    def __init__(self, channels, context_dim, num_head_channels, dropout=0., T=None, **kwargs):
        super().__init__()
        self.spatial_attention = SpatialTransformer(
            T=T,
            context_dim=context_dim,
            channels=channels, num_head_channels=num_head_channels,
            dropout=dropout
        )
        model_channels = kwargs.get('model_channels', 64)
        self.temporal_attention = TemporalTransformer(
            T=T,
            context_dim=None,
            channels=channels, num_head_channels=num_head_channels,
            dropout=dropout,
            model_channels=model_channels
        )
    def forward(self, x, temb, context=None, is_im=False, **kwargs):
        BT, C, H, W = x.shape
        x = self.spatial_attention(x, context = context)
        x = self.temporal_attention(x, is_im=is_im)
        return x


class SDUNetModel3D(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        context_dim=None,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_head_channels=64,
        use_scale_shift_norm=False,
        num_frames=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        default_attn_args = {
            "context_dim": context_dim,
            "num_head_channels": num_head_channels,
            "dropout": dropout,
            "time_embed_dim": time_embed_dim,
            "use_checkpoint": use_checkpoint,
            "use_scale_shift_norm": use_scale_shift_norm,
            "dims": dims,
            "T": num_frames,
            "model_channels": model_channels,
        }
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, out_channels=mult * model_channels, **default_attn_args)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedTransformer(ch, **default_attn_args)
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        
        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(ch, **default_attn_args),
            FactorizedTransformer(ch, **default_attn_args),
            ResBlock(ch, **default_attn_args),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        out_channels=model_channels * mult,
                        **default_attn_args
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedTransformer(
                            ch, **default_attn_args,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            Pseudo3DConv(model_channels, out_channels, 3, num_frames)
        )
        if kwargs.get("pretrain", None):
            path = kwargs["pretrain"]
            state_dict = torch.load(path, map_location="cpu")
            frozen_set = set(state_dict.keys())
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            frozen_set -= set(unexpected_keys)
            _state_dict = {}
            for k in unexpected_keys:
                ends = k.split(".")[-1]
                _k = k.replace(f".{ends}", f".spatial_conv.{ends}")
                _state_dict[_k] = state_dict[k]
            frozen_set = frozen_set.union(set(_state_dict.keys()))
            m, u = self.load_state_dict(_state_dict, strict=False)
            miss = set(m) - frozen_set
            assert len(u) == 0, f"Unmatched keys: {u}"
            

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype
        
    def forward_with_cond_scale(self, *args, cond_scale=2, **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, **kwargs.pop("cond"))
        return null_logits + (logits - null_logits) * cond_scale
        
    def forward(self, x, timesteps, cond= None, is_im=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        B, C, T, H, W = x.shape
        if cond is not None:
            if isinstance(cond, th.Tensor):
                context = cond
            else:
                context = cond["text"]
        timesteps = timesteps.view(B, 1).expand(B, T)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        timesteps = timesteps.reshape(B*T)
        hs = []

        emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)
        h = x.type(self.inner_dtype)
        args = {
            "context": context if cond is not None else None,
            "temb": emb,
            "is_im": is_im,
        }
        for layer, module in enumerate(self.input_blocks):
            h = module(h, emb,  **args)
            hs.append(h)
        h = self.middle_block(h, emb,  **args)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb,  **args)
        h = h.type(x.dtype)
        out = self.out(h)
        out = rearrange(out, '(b t) c h w -> b c t h w', b=B)
        return out

if __name__ == "__main__":
    model = SDUNetModelV10_2_Mask(image_size=32, in_channels=4, model_channels=320, 
            out_channels=4, num_res_blocks=2,
            attention_resolutions=(1, 2, 4), 
            channel_mult=[1, 2, 4, 4], 
            context_dim=1024,
            num_head_channels=64,
            num_frames=16,
            pretrain="pretrain/unet_v2.ckpt")
    x = th.randn(1, 4, 16, 32, 32).cuda()
    context = th.randn(1, 72, 1024).cuda()
    timesteps = th.Tensor([1]).cuda()
    model.cuda()
    model.eval()
    out = model(x, timesteps, cond=context, is_im=True)
    print(out.shape)