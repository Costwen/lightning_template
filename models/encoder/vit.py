import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from math import log, pi

def exists(val):
    return val is not None

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, rot_emb = None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q, k = apply_rot_emb(q, k, rot_emb)
        # expand cls token keys and values across time or space and concat
        # attention
        out = attn(q, k, v, mask = mask)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rot_emb(q, k, rot_emb):
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

class ViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        image_size = 256,
        patch_size = 8,
        channels = 3,
        depth = 8,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True,
        shift_tokens = False,
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.use_rotary_emb = rotary_emb

        if rotary_emb:
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            spatial_attn, ff = map(lambda t: PreNorm(dim, t), ( spatial_attn, ff))

            self.layers.append(nn.ModuleList([spatial_attn, ff]))


    def forward(self, image, frame_mask = None):
        b, _, h, w, device, p = *image.shape, image.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of image must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)
        hp, wp = (h // p), (w // p)
        n = hp * wp

        # image to patch embeddings
        image = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.to_patch_embedding(image)

        # positional embedding
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        for (spatial_attn, ff) in self.layers:
            x = spatial_attn(x, rot_emb = image_pos_emb) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTNerf(nn.Module):
    def __init__(self, dim = 512, channels = 3, patch_size = 8, image_size = 256, depth = 8, heads = 8, dim_head = 64, mlp_dim = 256, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.patch_size = patch_size
        self.encoder = ViTEncoder(dim = dim, channels = channels, patch_size = patch_size, image_size = image_size, depth = depth, heads = heads, dim_head = dim_head, ff_dropout = dropout)
        self.x_token = nn.Parameter(torch.randn(1, 1, dim))
        self.y_token = nn.Parameter(torch.randn(1, 1, dim))

        self.x_pos_embedding = nn.Parameter(torch.randn(1, image_size // patch_size + 1, dim))
        self.y_pos_embedding = nn.Parameter(torch.randn(1, image_size // patch_size + 1, dim))
        self.x_quant_att = Transformer(dim, depth = 2, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim)
        self.y_quant_att = Transformer(dim, depth = 2, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim)

        self.pre_x = nn.Conv1d(dim, dim, 1)
        self.pre_y = nn.Conv1d(dim, dim, 1)
        self.post_x = nn.Conv1d(dim, dim, 1)
        self.post_y = nn.Conv1d(dim, dim, 1)

        self.nerf = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Softplus(),
            nn.Linear(dim, dim),
            nn.Softplus(),
            nn.Linear(dim, 3),
        )


    def forward(self, image, frame_mask = None):
        b, _, h, w, device, p = *image.shape, image.device, self.patch_size
        hp, wp = (h // p), (w // p)
        x = self.encoder(image, frame_mask = frame_mask)
        # x: b (h w) d
        h_x = rearrange(x, 'b (h w) d-> (b w) h d', h = hp)
        # 使用一个token 获取对应x轴的所有y轴的信息
        x_token = repeat(self.x_token, '1 1 d -> bw 1 d', bw = h_x.size(0))

        h_x = torch.cat([h_x, x_token], dim=1)
        h_x += self.x_pos_embedding[:, :hp+1]
        
        # transformer
        h_x = self.x_quant_att(h_x)[:, -1]
        hx = rearrange(h_x, '(b w) d -> b d w', b = b)
        # just a projection
        hx = self.pre_x(hx)
        hx = torch.tanh(hx)
        hx = self.post_x(hx)

        h_y = rearrange(x, 'b (h w) d-> (b h) w d', w = wp)
        y_token = repeat(self.y_token, '1 1 d -> bh 1 d', bh = h_y.size(0))
        h_y = torch.cat([h_y, y_token], dim=1)
        h_y += self.y_pos_embedding[:, :wp+1]
        h_y = self.y_quant_att(h_y)[:, -1]
        hy = rearrange(h_y, '(b h) d -> b d h', b = b)

        hy = self.pre_y(hy)
        hy = torch.tanh(hy)
        hy = self.post_y(hy)

        # 构建特征图 [b, dim // 4, hp, wp]
        plane = hy.unsqueeze(3) + hx.unsqueeze(2)
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"), dim=-1).to(device)
        coords = repeat(coords, 'h w c -> b h w c', b = b)
        features = torch.nn.functional.grid_sample(plane, coords, mode='bilinear', align_corners=True)
        features = rearrange(features, 'b c h w -> b h w c')
        rgb = self.nerf(features)
        rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # mlpnerf
        rgb = rearrange(rgb, 'b h w c -> b c h w')
        return rgb

if __name__ == '__main__':
    b, c, h, w = 1, 3, 256, 256
    image = torch.randn(b, c, h, w).cuda()
    model = ViTNerf().cuda()
    out = model(image)