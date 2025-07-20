# predict.py – SDXL → (opcional) Latent×2 → (opcional) SD-X4
from pathlib import Path
from typing import List
import functools, torch, PIL.Image as Image
from cog import BasePredictor, Input, Path as CogPath
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionX4UpscalePipeline,
)

def load_pipe(model_id: str, cls, tiling=False):
    pipe = cls.from_pretrained(model_id, torch_dtype=torch.float16,
                               low_cpu_mem_usage=True)
    if tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    if torch.cuda.is_available():
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        pipe.to(memory_format=torch.channels_last)
    return pipe

@functools.lru_cache(maxsize=128)
def cached_embed(pipe, text):
    tok = pipe.tokenizer(text, return_tensors="pt").to(pipe.device)
    return pipe.text_encoder(**tok)[0]

MAX_X4_IN = 1024  # px  (seguro contra OOM)

class Predictor(BasePredictor):
    def setup(self):
        self.base  = load_pipe("stabilityai/stable-diffusion-xl-base-1.0",
                               StableDiffusionXLPipeline)
        self.lat2x = load_pipe("stabilityai/sd-x2-latent-upscaler",
                               StableDiffusionLatentUpscalePipeline)
        self.x4    = load_pipe("stabilityai/sd-x4-upscaler",
                               StableDiffusionX4UpscalePipeline, tiling=True)

    def predict(
        self,
        prompt: str = Input(description="Texto del prompt"),
        width: int  = Input(default=768, ge=512, le=1024),
        height: int = Input(default=768, ge=512, le=1024),
        steps: int  = Input(default=24,  ge=10,  le=40),
        guidance: float = Input(default=5.0, ge=1, le=20),
        upscale_mode: str = Input(
            default="combo_x2_then_x4",
            choices=["none", "latent_x2", "sd_x4", "combo_x2_then_x4"]),
        up_steps: int = Input(default=12, ge=6, le=30),
    ) -> List[CogPath]:

        base = self.base(
            prompt, width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance,
            prompt_embeds=cached_embed(self.base, prompt),
        ).images[0]
        if upscale_mode == "none":
            p = Path("base.png"); base.save(p); return [CogPath(p)]

        img2 = base
        if upscale_mode in {"latent_x2", "combo_x2_then_x4"}:
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            img2 = self.lat2x(
                prompt=prompt, image=base,
                num_inference_steps=up_steps, guidance_scale=guidance,
            ).images[0]
            if upscale_mode == "latent_x2":
                p = Path("lat2x.png"); img2.save(p); return [CogPath(p)]

        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        if max(img2.size) > MAX_X4_IN:
            scale = MAX_X4_IN / max(img2.size)
            img2 = img2.resize((int(img2.width*scale), int(img2.height*scale)),
                               Image.LANCZOS)

        img4 = self.x4(
            prompt=prompt, image=img2,
            num_inference_steps=up_steps, guidance_scale=guidance,
        ).images[0]
        p = Path("out4k.png"); img4.save(p)
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        return [CogPath(p)]
