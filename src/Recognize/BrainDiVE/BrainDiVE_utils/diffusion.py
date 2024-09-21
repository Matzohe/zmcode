import torch
import torch.nn as nn
import diffusers
from diffusers import StableDiffusionPipeline
import math
import torch.nn.functional as F
from PIL import Image
import numpy as np

from .CrossAttnStoreProcessor import CrossAttnStoreProcessor

class DiffusionPipe(nn.Module):
    # Special designed for BrainDiVE
    def __init__(self, config):
        super().__init__()
        self.device = config.INFERENCE["device"]
        self.sag_scale = float(config.BRAINDIVE["sag_scale"])
        self.time_step = int(config.DIFFUSION["time_step"])
        self.model = StableDiffusionPipeline.from_pretrained(
            config.DIFFUSION["model_name"]).to(config.INFERENCE["device"])
        
        # Inference time parameter setting
        self.scheduler = self.model.scheduler
        self.scheduler.set_timesteps(self.time_step)

        # different model parts for building diffusion models
        self.vae = self.model.vae
        self.vae.to(device=self.device)
        self.unet = self.model.unet
        self.unet = self.unet.to(device=self.device)
        self.text_encoder = self.model.text_encoder
        self.text_encoder = self.text_encoder.to(device=self.device)

        self.tokenizer = self.model.tokenizer
        self.image_size = int(config.DIFFUSION["image_size"])

        self.store_processor = CrossAttnStoreProcessor()
        self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = self.store_processor

    def TextEmbedding(self, prompt):
        tokenizedinfo = self.tokenizer(prompt, return_tensors="pt").input_ids
        tokenizedinfo = tokenizedinfo.to(device=self.device)
        return self.text_encoder(tokenizedinfo)[0]
    
    def NoiseProcess(self, latents, text_embeddings, timesteps):
        latent_model_input = self.scheduler.scale_model_input(latents, timesteps)
        noise_pred = self.unet(
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                    ).sample
        pred_x0 = self.pred_x0(latents, noise_pred, timesteps)
        cond_attn = self.store_processor.attention_probs
        degraded_latents = self.sag_masking(
            pred_x0, cond_attn, timesteps, self.pred_epsilon(latents, noise_pred, timesteps)
        )
        degraded_pred = self.unet(degraded_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        noise_pred += self.sag_scale * (noise_pred - degraded_pred)
        return noise_pred

    def DiffusionProcess(self, latents, noise_pred, timesteps):
        return self.scheduler.step(noise_pred, timesteps, latents).prev_sample
    
    def NoiseInitialize(self):
        return torch.randn((1, self.unet.in_channels, self.image_size // 8, self.image_size // 8), device=self.device, requires_grad=False)
    
    def ImageEncoding(self, image):
        with torch.no_grad():
            return self.vae.encode(image).latent_dist.sample
    
    def ImageDecoding(self, latents):
        with torch.no_grad():
            latents = 1 / self.vae.config.scaling_factor * latents
            return self.vae.decode(latents).sample
    
    def sag_masking(self, original_latents, attn_map, t, eps):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.attention_head_dim
        if isinstance(h, list):
            h = h[-1]
        map_size = math.isqrt(hw1)

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size, map_size).unsqueeze(1).repeat(1, latent_channel, 1, 1).type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = self.gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # Noise it again to match the noise level
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

        return degraded_latents
    
    def gaussian_blur_2d(self, img, kernel_size, sigma):
        if not hasattr(self, "kernel2d"):
            ksize_half = (kernel_size - 1) * 0.5

            x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

            pdf = torch.exp(-0.5 * (x / sigma).pow(2))

            x_kernel = pdf / pdf.sum()
            x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

            kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
            kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
            self.kernel2d = kernel2d
        kernel2d = self.kernel2d
        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        img = F.pad(img, padding, mode="reflect")
        img = F.conv2d(img, kernel2d, groups=img.shape[-3])

        return img
    
    def pred_x0(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            # predict V
            model_output = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_original_sample
    
    def pred_epsilon(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_eps = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_eps = (sample - (alpha_prod_t ** 0.5) * model_output) / (beta_prod_t ** 0.5)
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_eps = (beta_prod_t ** 0.5) * sample + (alpha_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )
        return pred_eps


def ImageOutput(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image