from torch import nn
import torch as th

def timestep_embedding(timesteps, dim, num_frames = None, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    if num_frames is not None:
        embedding = repeat(embedding, 'b d -> (b t) d', t=num_frames)
    return embedding


class CoordinatesEMB(nn.Module):
    def __init__(self, hidden_size, N, T):
        super().__init__()
        
        self.N = N
        self.T = T
        self.linear1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        n, _, h, w = x.shape
        # cache
        if not hasattr(self, 'coords'):
            coords_W = th.linspace(-1, 1, steps = h).view(1, -1, 1, 1).repeat(n, 1, w, 1)
            coords_H = th.linspace(-1, 1, steps = w).view(1, 1, -1, 1).repeat(n, h, 1, 1)
            coords = th.cat((coords_h, coords_w), dim=-1).to(x.device)
            coords_T = th.linspace(-1, 1, steps = self.diffusion_timesteps).view(-1, 1, 1, 1, 1).repeat(1, self.video_timesteps, 1, 1, 1)
            coords_N = th.linspace(-1, 1, steps = self.video_timesteps).view(1, -1, 1, 1, 1).repeat(self.diffusion_timesteps, 1, 1, 1, 1)
            new_coords = th.cat((coords_dt, coords_vt), dim=-1).to(x.device)
            self.register_buffer('coords', coords)
            self.register_buffer('new_coords', new_coords)
        
        coords = th.cat((self.new_coords[dt, vt].repeat(1, h, w, 1), self.coords), dim=-1)

        fourier_features = self.linear_1(th.sin(self.linear_0(coords.view(-1, 4))))
        fourier_features = einops.rearrange(fourier_features, '(n h w) c -> n c h w', h=h, w=w)
        scale, shift = th.chunk(fourier_features, 2, dim=1)
        x = self.out_norm(x) * (1 + scale) + shift
        x = self.out_rest(x)
        return x