
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from models.op.common import checkpoint, Normalize, zero_module, default, exists, GEGLU
from einops_exts import check_shape, rearrange_many
from models.op.pos_emb import *

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
        # weight = torch.stack([
        #     torch.concat([torch.linspace(2.0/((1+t)*t), 2.0/(1+t), steps=t), torch.zeros(((T-t)))]) \
        #     for t in range(1, T+1)])

        # self.weight = nn.Parameter(
        #     weight, requires_grad=True
        # )

    def forward(self, x, context=None, mask=None):

        h = self.heads
        T = self.T
        q = self.to_q(x)

        if context is not None:
            context = repeat(context, 'b n d -> (b t) n d', t=T)
        
        # if context is None:
        #     _, n, d = x.shape
        #     fuse = rearrange(x, "(b t) n d -> b t (n d)", t=T)
        #     fuse = self.weight @ fuse
        #     fuse = rearrange(fuse, "b t (n d) -> (b t) n d", n=n,d=d)
        #     context = torch.cat([x, fuse], dim=1) # (b t) 2n d

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)




class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False, T=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,T=T)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,T=T)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
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
    def __init__(self, channels, num_heads,
                 depth=1, dropout=0., context_dim=None, checkpoint=False,T=None):
        super().__init__()
        self.channels = channels
        inner_dim = channels
        n_heads = num_heads
        d_head = inner_dim // n_heads
        self.norm = Normalize(channels)
        self.T = T
        self.proj_in = nn.Conv2d(channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=checkpoint, T=T)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False, T=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,T=T)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,T=T)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.T = T
        self.msg_ln = nn.LayerNorm(dim)
        self.msg_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.checkpoint)

    def _forward(self, x, context=None):
        # (b t) (1 + hw) d
        msg_token = x[:, 0, :] # (b t) d
        msg_token = rearrange(msg_token, '(b t) d -> b t d', t=self.T)
        msg_token = msg_token + self.msg_attn(self.msg_ln(msg_token), self.msg_ln(msg_token), self.msg_ln(msg_token), need_weights=False)[0]
        msg_token = rearrange(msg_token, 'b t d -> (b t) 1 d')
        x = torch.cat([msg_token, x], dim=1)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = x[:, 1:, :]
        x = self.ff(self.norm3(x)) + x
        return x


class CrossSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, channels, num_heads,
                 depth=1, dropout=0., context_dim=None, checkpoint=False,T=None):
        super().__init__()
        self.channels = channels
        inner_dim = channels
        n_heads = num_heads
        d_head = inner_dim // n_heads
        self.norm = Normalize(channels)
        self.T = T
        self.proj_in = nn.Conv2d(channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [CrossTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=checkpoint, T=T)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        scale = channels ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(channels))
        

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x

        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, 'b c h w -> b (h w) c')

        emb = repeat(self.class_embedding, "c -> b 1 c", b = b)
        x = torch.cat([emb, x], dim=1)  # b (h*w+1) c

        for block in self.transformer_blocks:
            x = block(x, context=context)

        x, cls = x[:, 1:, :], x[:, 0, :]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in




class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, 
    T=None, checkpoint=False, **kwargs):
        super().__init__()
        self.attn1 = TemporalAttention(channels=dim, num_heads=n_heads,dropout=dropout, T=T, **kwargs)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = TemporalAttention(channels=dim, num_heads=n_heads,dropout=dropout, T=T, **kwargs)  # is self-attn if context is none
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, femb, attn_mask):
        return checkpoint(self._forward, (x,femb, attn_mask), self.checkpoint)

    def _forward(self, x, femb, attn_mask):
        # x = self.attn1(self.norm1(x), femb, attn_mask=attn_mask) + x
        # x = self.attn2(self.norm2(x), femb, attn_mask=attn_mask) + x
        x = self.attn1(x, attn_mask)
        x = self.attn2(x, attn_mask)
        x = self.ff(self.norm3(x)) + x
        return x

class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, channels, num_heads,
                 depth=1, dropout=0., T=None, checkpoint=False, **kwargs):
        super().__init__()
        self.channels = channels
        inner_dim = channels
        n_heads = num_heads
        d_head = inner_dim // n_heads
        self.norm = Normalize(channels)

        self.proj_in = nn.Conv1d(channels,
                                inner_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.transformer_blocks = nn.ModuleList(
            [TemporalTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,\
                T=T, checkpoint=checkpoint, **kwargs)
            for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                              channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, femb, attn_mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        B, D, C, T = x.shape
        x = rearrange(x, 'b d c t -> (b d) c t')
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x, femb, attn_mask=attn_mask)
        x = rearrange(x, 'b t c -> b c t')
        x = self.proj_out(x)
        x = rearrange(x, '(b d) c t -> b d c t', b=B, d=D)
        return x + x_in
    def finetune(self):
        self.requires_grad_(True)

class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dim_head=64, dropout=0., T = None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels)
        self.T = T
        self.norm = nn.LayerNorm(channels)
        # self.pos_bias = LearnableAbsolutePositionEmbedding(T, channels)
        self.pos_bias = RelativePositionEmbedding(2*T-1, num_attention_heads=num_heads, hidden_size=channels, \
                    position_embedding_type='contextual(2)')

    def forward(self, x, mask=None):
        pos_bias = self.pos_bias
        h = self.num_heads
        _x = x
        if pos_bias.is_absolute:
            x = pos_bias(x)
        x = self.norm(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        sim = torch.einsum('b h m d, b h n d -> b h m n', q, k)

        if pos_bias is not None and pos_bias.is_absolute is False:
            bias = pos_bias.compute_bias(q, k, self.to_q, self.to_k)
            sim = sim + bias

        sim = sim * self.scale

        if exists(mask):
            mask = repeat(mask, '1 i j -> b 1 i j', b=sim.shape[0])
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h m n, b h n d -> b h m d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.proj_out(out)
        return out + _x

    def finetune(self):
        self.requires_grad_(True)


class RPENet(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        self.silu = nn.SiLU()
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, temb, relative_distances):
        distance_embs = torch.stack(
            [torch.log(1+(relative_distances).clamp(min=0)),
             torch.log(1+(-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1
        )  # BxTxTx3
        B, T, _ = relative_distances.shape
        C = self.channels
        emb = self.embed_diffusion_time(temb).view(B, T, 1, C) \
            + self.embed_distances(distance_embs)  # B x T x T x C
        return self.out(self.silu(emb)).view(*relative_distances.shape, self.num_heads, self.channels//self.num_heads)


class RPE(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self, channels, num_heads, time_embed_dim, T=None, use_rpe_net=False):
        """ This module handles the relative positional encoding.
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)
        else:
            self.lookup_table_weight = nn.Parameter(
                torch.zeros(2 * T - 1,
                         self.num_heads,
                         self.head_dim))

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            return self.lookup_table_weight[pairwise_distances]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, temb, mode):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances, temb)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances, temb)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances, temb):
        # qv is either of q or k and has shape BxDxHxTx(C/H)
        # Output shape should be # BxDxHxTxT
        R = self.get_R(pairwise_distances, temb)
        return torch.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
            "bdhtf,btshf->bdhts", qk, R  # BxDxHxTxT
        )

    def forward_v(self, attn, pairwise_distances, temb):
        # attn has shape BxDxHxTxT
        # Output shape should be # BxDxHxYx(C/H)
        R = self.get_R(pairwise_distances, temb)
        return torch.einsum(  # See Eq. 16ish in https://arxiv.org/pdf/2107.14222.pdf
            "bdhts,btshf->bdhtf", attn, R  # BxDxHxTxT
        )

class RPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(self, channels, num_heads, use_checkpoint=False,
                 time_embed_dim=None, use_rpe_net=True,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True,
                 T = None,
                 **kwargs
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels)
        self.T = T
        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None

        def make_rpe_func():
            return RPE(
                channels=channels, num_heads=num_heads,
                time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
                T=T
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, temb, frame_indices, attn_mask=None):
        out = self._forward(x, temb, frame_indices, attn_mask)
        return out

    def _forward(self, x, temb, frame_indices, attn_mask=None):
        # x [B, D, T, C]
        B, D, T, C = x.shape
        _x = x
        x = self.norm(x)
        h = self.num_heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b d t (h c) -> b d h t c', h=h), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) # BxDxHxTxT

        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2)) # BxTxT

        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, temb=temb, mode="qk")

        if self.rpe_q is not None:
            attn += self.rpe_q(k, pairwise_distances, temb=temb, mode="qk").transpose(-1, -2)
        
        attn = attn * self.scale

        if exists(attn_mask):
            attn_mask = repeat(attn_mask, '1 i j -> b 1 1 i j', b=attn.shape[0])
            max_neg_value = -torch.finfo(attn.dtype).max
            attn.masked_fill_(~attn_mask, max_neg_value)

        attn = attn.softmax(dim=-1)

        out = attn @ v

        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, temb=temb, mode="v")
    
        out = torch.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)

        out = self.proj_out(out) + _x
        return x

    def finetune(self):
        self.requires_grad_(True)