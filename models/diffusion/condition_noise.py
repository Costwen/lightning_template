import torch
from torch import nn, einsum
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from functools import partial
from models.op.common import *
# from  https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/upscaling.py
class AbstractLowScaleModel(nn.Module):
    # for concatenating a downsampled image to the latent representation
    def __init__(self, noise_schedule_config=None):
        super(AbstractLowScaleModel, self).__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x):
        return x, None

    def decode(self, x):
        return x


class ConditionNoise(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def forward(self, x, noise_level=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        z = self.q_sample(x, noise_level)
        return z

if __name__ == "__main__":
    video = torch.randn((1, 3, 16, 32, 32)).cuda()
    model = ConditionNoise(
        noise_schedule_config={
            "linear_start": 0.0001,
            "linear_end": 0.02,
        },
        max_noise_level=350
    ).cuda()
    output, noise_level = model(video)
    print(output.shape, noise_level)
#       params:
#         noise_schedule_config: # image space
#           linear_start: 0.0001
#           linear_end: 0.02
#         max_noise_level: 350

