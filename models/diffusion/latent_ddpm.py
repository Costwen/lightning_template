import torch
from torch import nn, einsum
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from functools import partial
import pytorch_lightning as pl
from models.op.common import *


class LatentDDPM(pl.LightningModule):
    def __init__(
        self,
        unet,
        num_frames,
        channels = 3,
        timesteps = 1000,
        beta_schedule = 'linear',
        loss_type = 'l1',
        parameterization = "eps",
        linear_start = 0.00085,
        linear_end = 0.0120,
        cond_drop_prob = 0.,
        autoencoder = None,
        text_encoder = None,
        **kwargs
    ):
        super().__init__()
        assert parameterization in ['eps', 'x0', 'v']
        self.parameterization = parameterization
        self.channels = channels
        self.num_frames = num_frames
        self.denoise_fn = unet
        self.cond_drop_prob = cond_drop_prob
        betas = make_beta_schedule(beta_schedule, timesteps,\
                                       linear_start=linear_start,
                                       linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # dynamic thresholding when sampling
        self.encoder = autoencoder
        self.text_encoder = text_encoder
        
        if kwargs.get("pretrain", None):
            load_pretrained_model(self, kwargs["pretrain"])
        
        self.condition_noise = kwargs.get("condition_noise", None)
        

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1., clip_x_start=False, **kwargs):
        out = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale, **kwargs)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else lambda x: x

        if isinstance(out, tuple):
            pred, mask, mae = out
            pred = mae.unpatchify(pred)
            out = rearrange(pred, 'b c h w -> b c 1 h w')

        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise = out)
        elif self.parameterization == 'x0':
            x_recon = out
        elif self.parameterization == 'v':
            v = out
            x_recon = self.predict_start_from_v(x, t, v)
            x_recon = maybe_clip(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale, **kwargs)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1., **kwargs):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale, **kwargs)

        return {
            "samples": unnormalize(img)
        }

    @torch.inference_mode()
    def sample(self, shape, cond = None, cond_scale = 1., batch_size = 16, **kwargs):
        return self.p_sample_loop(shape, cond = cond, cond_scale = cond_scale, **kwargs)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x_start

        if self.loss_type == 'l1':
            loss = F.l1_loss(target, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, x_recon)
        else:
            raise NotImplementedError()

        return loss
    def get_text_encode(self, text):
        return self.text_encoder.encode(text)

    def get_video_encode(self, video):
        T = video.shape[2]
        video = rearrange(video, 'b c t h w -> (b t) c h w')
        encode = self.encoder.get_scale_encoding(video)
        encode = rearrange(encode, '(b t) c h w -> b c t h w', t = T)
        return encode

    def get_video_decode(self, encode):
        T = encode.shape[2]
        encode = rearrange(encode, 'b c t h w -> (b t) c h w')
        video = self.encoder.get_scale_decoding(encode)
        video = rearrange(video, '(b t) c h w -> b c t h w', t = T)
        return video

    def forward(self, x, cond = None, *args, **kwargs):
        x = self.get_video_encode(x)
        b, t, device, = x.shape[0], x.shape[2] ,x.device

        has_cond = self.cond_drop_prob < np.random.rand()

        is_im = kwargs.get("is_im", False)

        # classifier-free
        if not has_cond and self.training:
            if is_im:
                cond["text"] = t * [ b*[""]]
            else:
                cond["text"] = b * [""]
        
        # joint training
        if is_im:
            cond_list = []
            for b_id in range(b):
                input_text = [cond["text"][i][b_id] for i in range(len(cond["text"]))]
                cond_list.append(self.get_text_encode(input_text))
            cond["text"] = torch.stack(cond_list)
            cond["text"] = rearrange(cond["text"], 'b t n c -> (b t) n c')
        else:
            cond["text"] = self.get_text_encode(cond["text"])
       
        # tsr:
        if cond.get("tsr", None) is not None:
            cond["tsr"] = x[:,:,0::4]
            if self.condition_noise:
                cond["tsr"] = self.condition_noise(cond["tsr"])

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, cond, *args, **kwargs)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
