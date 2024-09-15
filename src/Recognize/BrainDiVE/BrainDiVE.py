import torch
import torch.nn as nn
import numpy as np
from diffusers import LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from BrainDiVE_utils.diffusion import DiffusionPipe
from ...MultiModal.clip import from_pretrained

class BrainDiVE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.INFERENCE["device"]
        self.pipe = DiffusionPipe(config)
        self.clip = from_pretrained.load(config.BRAINDIVE("clip_model"), self.device)
        self.clip_guidance_scale = float(config.BRAINDIVE("clip_guidance_scale"))
        self.weight = None
        self.roi = None

    def load_weight(self, path):
        self.weight = torch.from_numpy(np.load(path)).to(self.device).to(dtype=torch.float32)
        
    def load_roi(self, path):
        self.roi = torch.from_numpy(np.loadtxt(path))
        self.weight = self.weight.T
        index = np.zeros(len(self.weight),dtype=bool)
        for i in range(len(index)):
            if self.roi[i] > 0:
                index[i] = True
        self.weight = self.weight[index][:]
        self.weight = self.weight.T

    def brainProcess(self, latent, timestep, current_step, text_embeddings, noise_orignal, clip_guidance_scale):
        latent_input = latent.detach().requires_grad_()
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_input, timestep)
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[current_step]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        if not hasattr(self, "OPENAI_CLIP_MEAN"):
            self.OPENAI_CLIP_MEAN = torch.from_numpy(
                np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[None, :, None, None]).to(image.device)
            self.OPENAI_CLIP_STD = torch.from_numpy(
                np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[None, :, None, None]).to(image.device)
            self.OPENAI_CLIP_MEAN.requires_grad = False
            self.OPENAI_CLIP_STD.requires_grad = False
        fp32_image = image.float()
        resized_image = torch.nn.functional.interpolate(fp32_image, size=224, mode="bilinear")
        normalized_image = (resized_image - self.OPENAI_CLIP_MEAN) / self.OPENAI_CLIP_STD

        image_embedding = self.clip.encode_image(normalized_image)
        image_embedding = image_embedding.to(dtype=torch.float32)
        fMRI_prediction = image_embedding @ self.weight
        loss = fMRI_prediction.sum() / self.roi.sum() * clip_guidance_scale

        grads = -torch.autograd.grad(loss, latents)[0]
        del loss
        del normalized_image
        del resized_image
        del fp32_image
        del image

        torch.cleanup()

        latents = latents.detach()
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma ** 2)
            noise_pred = noise_orignal
        else:
            noise_pred = noise_orignal - torch.sqrt(beta_prod_t) * grads
        return noise_pred, latents
    
    def process(self, latent, timestep, current_step, text_embeddings):
        noise_pred = self.pipe.NoiseProcess(latent, text_embeddings, timestep)
        noise_pred, latent = self.brainProcess(latent, timestep, current_step, text_embeddings, noise_pred, self.clip_guidance_scale)
        latent = self.pipe.scheduler.step(noise_pred, timestep, latent).prev_sample
        return latent
    
    def forward(self):
        latent = self.pipe.NoiseInitialize()
        text_embeddings = self.pipe.TextEmbedding("")
        for current_step, timestep in enumerate(self.pipe.scheduler.timesteps):
            latent = self.process(latent, timestep, current_step, text_embeddings)
        return latent

    