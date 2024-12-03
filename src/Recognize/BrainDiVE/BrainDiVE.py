import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import numpy as np
from diffusers import LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from tqdm import tqdm
import os
from PIL import Image
from .BrainDiVE_utils.diffusion import DiffusionPipe, ImageOutput
from ...MultiModal.clip.from_pretrained import load

class BrainDiVE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.INFERENCE["device"]
        self.pipe = DiffusionPipe(config)
        self.clip, _ = load(config.BRAINDIVE["clip_model"], self.device)
        self.clip_guidance_scale = float(config.BRAINDIVE["clip_guidance_scale"])
        self.weight = None
        self.bias = None
        self.roi = None
        self.before_weight = None
        self.middle_image_save_root = config.BRAINDIVE["middle_image_save_root"]
        self.middle_image_save_name = config.BRAINDIVE["middle_image_save_name"]
        self.grad_save_root = config.BRAINDIVE["middle_image_grad_save_root"]
        self.grad_save_name = config.BRAINDIVE["middle_image_grad_save_name"]


    def load_weight(self, path):
        self.weight = torch.load(path, map_location=self.device)['weight'].to(dtype=torch.float32)
        self.bias = torch.load(path, map_location=self.device)['bias'].to(dtype=torch.float32)
        self.before_bias = torch.load(path, map_location=self.device)['bias'].to(dtype=torch.float32)
        self.before_weight = torch.load(path, map_location=self.device)['weight'].to(dtype=torch.float32)
        # self.weight = torch.from_numpy(np.load(path)).to(self.device).to(dtype=torch.float32)
        # self.before_weight = torch.from_numpy(np.load(path)).to(self.device).to(dtype=torch.float32)
        

    def load_roi(self, path):
        self.roi = torch.from_numpy(np.loadtxt(path)).to(dtype=torch.float32).to(device=self.device)
        self.weight = self.before_weight
        self.bias = self.before_bias
        index = np.zeros(len(self.weight),dtype=bool)
        for i in range(len(index)):
            if self.roi[i] > 0:
                index[i] = True
        self.weight = self.weight[index][:]
        self.bias = self.bias[index]
        self.weight = self.weight.T


    def brainProcess(self, latent, timestep, current_step, text_embeddings, noise_orignal, clip_guidance_scale):
        latent_input = latent.detach().requires_grad_()
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_input, timestep)
        noise_pred = self.pipe.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.pipe.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler)):
            alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latent_model_input - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latent_model_input * (1 - fac)
        elif isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            sigma = self.pipe.scheduler.sigmas[current_step]
            sample = latent_model_input - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.pipe.scheduler)} not supported")

        sample = 1 / self.pipe.vae.config.scaling_factor * sample
        image = self.pipe.vae.decode(sample).sample
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
        image_embedding = image_embedding / torch.norm(image_embedding, dim=1, p=2)
        fMRI_prediction = image_embedding @ self.weight + self.bias
        loss = fMRI_prediction.sum() / self.roi.sum() * clip_guidance_scale
        grads = -torch.autograd.grad(loss, latent_input, retain_graph=True)[0]
        image_grads = -torch.autograd.grad(loss, normalized_image)[0]
        image_grads = ((torch.abs(image_grads) / torch.max(torch.abs(image_grads))).clamp(0, 1).data * 255).to(dtype=torch.uint8).detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        del loss
        del latent_model_input
        del latent_input
        del normalized_image
        del resized_image
        del fp32_image
        del image

        latent = latent.detach()
        if isinstance(self.pipe.scheduler, LMSDiscreteScheduler):
            latent = latent + grads.data * (sigma ** 2)
            noise_pred = noise_orignal
        else:
            noise_pred = noise_orignal - torch.sqrt(beta_prod_t) * grads.data
        return noise_pred, latent, image_grads
    

    def process(self, latent, timestep, current_step, text_embeddings, 
                save_middle_grad=True, save_middle_image=True,
                subj_id=1, roi="FFA", fig_id=1):
        with torch.no_grad():
            noise_pred = self.pipe.NoiseProcess(latent, text_embeddings, timestep)
        noise_pred, latent, image_grads = self.brainProcess(latent, timestep, current_step, text_embeddings, noise_pred, self.clip_guidance_scale)
        latent = self.pipe.scheduler.step(noise_pred, timestep, latent).prev_sample
        if save_middle_grad:
            save_name = self.grad_save_root.format(subj_id, roi, fig_id)
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            save_root = os.path.join(save_name, self.grad_save_name.format(timestep))
            image = Image.fromarray(image_grads)
            image.save(save_root)
        if save_middle_image:
            save_name = self.middle_image_save_root.format(subj_id, roi, fig_id)
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            save_root = os.path.join(save_name, self.middle_image_save_name.format(timestep))
            image = self.pipe.ImageDecoding(latent)
            image = ImageOutput(image)
            image.save(save_root)
        return latent
    
    def forward(self, subj_id=1, roi="FFA", save_middle_image=True, save_middle_grad=True, fig_id=1):
        latent = self.pipe.NoiseInitialize()
        text_embeddings = self.pipe.TextEmbedding("")
        for current_step, timestep in tqdm(enumerate(self.pipe.scheduler.timesteps)):
            latent = self.process(latent, timestep, current_step, text_embeddings, save_middle_grad=save_middle_grad, 
                                  save_middle_image=save_middle_image, subj_id=subj_id, roi=roi,
                                  fig_id=fig_id)
        return latent

    