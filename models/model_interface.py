import torch
import torchvision as tv
import pytorch_lightning as pl
from sample.dpm_solver import DPMSolverSampler
from einops import rearrange
import os
import clip
from torchvision import transforms
import torch.nn.functional as F
from lion_pytorch import Lion
class ModelInterface(pl.LightningModule):
    def __init__(self, diffusion, **kwargs):
        super().__init__()

        self.diffusion = diffusion
        self.sampler = DPMSolverSampler(diffusion)
        self.sample_steps = kwargs.get("sample_steps", 50)
        self.scale = kwargs.get("scale", 7.5)
        self.exp_name = kwargs.get("exp_name", "default")
        self.image_lambda = kwargs.get("image_lambda", 0.3)
        self.optimizer = kwargs.get("optimizer", "adamw")

    def forward(self, x, cond = None, is_im=False):
        return self.diffusion(x, cond = cond, is_im=is_im)

    def validation_step(self, batch, batch_idx):
        video = batch['video']
        cond = batch['text']
        id_list = batch['id']
        x_samples = self.sample(video, cond)
        self._save_video(x_samples, cond, id_list)

    def training_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        x = batch['video']
        text = batch['text']
        tsr = batch.get('tsr', None)
        
        loss = self(x, cond = {'text': text, 'tsr': tsr})
        self.log("video_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # joint training with image
        if "images" in batch:
            images = batch['images']
            image_cond = batch['image_texts']
            image_loss = self(images, cond = {'text': image_cond}, is_im=True)
            loss = loss + self.image_lambda * image_loss
            self.log("image_loss", image_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log("total_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def _save_video(self, samples, cond, id_list):
        samples = samples.cpu()
        try:
            global_step = self.step
        except:
            global_step = self.global_step
        save_dir = os.path.join("outputs", self.exp_name, str(global_step))
        os.makedirs(save_dir, exist_ok=True)
        for video, text, idx in zip(samples, cond, id_list):
            save_path = os.path.join(save_dir, f"{idx}.mp4")
            tv.io.write_video(save_path, video, fps=3)

    def sample(self, video, cond):
        batch_size, num_frames, image_size = video.shape[0], video.shape[2], video.shape[3]
        shape = [4, num_frames, image_size//8, image_size//8]
        sampler = self.sampler
        
        uc = self.diffusion.get_text_encode(batch_size * [""])
        c = self.diffusion.get_text_encode(cond)
        samples = sampler.sample(S=self.sample_steps,
                                conditioning=c,
                                batch_size=batch_size,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=self.scale,
                                unconditional_conditioning=uc)
        x_samples = self.diffusion.get_video_decode(samples)
        x_samples = torch.clamp((x_samples * 0.5 + 0.5), 0, 1)
        x_samples = rearrange(x_samples, "b c t h w -> b t h w c")
        x_samples = (x_samples * 255).type(torch.uint8)
        return x_samples

    def test_step(self, batch, batch_idx):
        video = batch['video']
        cond = batch['text']
        id_list = batch['id']
        x_samples = self.sample(video, cond)
        self._save_video(x_samples, cond, id_list)
        return None
        
    def configure_optimizers(self):
        lr = self.lr
        params = list(self.diffusion.parameters())
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(params, lr=lr)
        elif self.optimizer == "lion":
            opt = Lion(params, lr = self.lr, weight_decay=1e-2)
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(opt, lr)
        return {
            'optimizer': opt,
            # 'lr_scheduler': lr_scheduler,
        }