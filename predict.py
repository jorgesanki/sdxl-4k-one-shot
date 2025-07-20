# predict.py  –  SD-XL base ➜ (opcional) Latent ×2 ➜ (opcional) SD-X4
# Requiere: diffusers ≥0.24, torch ≥2.1 (fp16) y tres checkpoints locales:
#   models/sdxl_base · models/latent_x2 · models/sd_x4

from pathlib import Path
from typing import List

import torch
from PIL import Image
from cog import BasePredictor, Input, Path as CogPath
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionLatentUpscalePipeline,
)

try:  # diffusers ≥0.27
    from diffusers import StableDiffusionX4UpscalePipeline as X4Pipe
except ImportError:  # diffusers <0.27
    from diffusers import StableDiffusionUpscalePipeline as X4Pipe  # type: ignore

# Resolución máxima admitida como input del x4 (evita OOM y artefactos)
MAX_X4_INPUT = 1024  # px


# -------------------------------------------------------------------------
# Helper ─ carga cualquier pipeline en fp16 + attention slicing + off-load
# -------------------------------------------------------------------------
def load_pipe(model_dir: str, cls, *, tiling: bool = False):
    pipe = cls.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()   # los bloques van/ vienen de GPU → RAM baja
    return pipe                       # ¡nada de .to("cuda")!


# -------------------------------------------------------------------------
class Predictor(BasePredictor):
    """SD-XL → (Latent ×2) → (SD-X4)  según el modo elegido"""

    # Se ejecuta una única vez al arrancar el contenedor
    def setup(self):
        self.pipe_base     = load_pipe("models/sdxl_base",   StableDiffusionXLPipeline)
        self.pipe_latent2x = load_pipe("models/latent_x2",   StableDiffusionLatentUpscalePipeline)
        self.pipe_x4       = load_pipe("models/sd_x4",       X4Pipe, tiling=True)

    # ---------------------- Inputs (todos con default) --------------------
    def predict(
        self,
        prompt: str = Input(description="Prompt principal"),
        width: int = Input(default=768,  ge=512, le=1024, description="Ancho imagen base"),
        height: int = Input(default=768, ge=512, le=1024, description="Alto imagen base"),
        steps: int = Input(default=24,   ge=10,  le=40,  description="Pasos SD-XL"),
        guidance: float = Input(default=5.0,  ge=1,  le=20, description="CFG scale (todos los pipes)"),
        upscale_mode: str = Input(
            default="none",
            choices=["none", "latent_x2", "sd_x4", "combo_x2_then_x4"],
            description="Upscaler a aplicar",
        ),
        up_steps: int = Input(default=12,  ge=6,  le=30, description="Pasos por upscaler"),
        latent_noise: int = Input(
            default=0,  ge=0,  le=100,
            description="noise_level para Latent ×2 (0 = sin confeti)",
        ),
        x4_noise: int = Input(
            default=0,  ge=0,  le=250,
            description="noise_level para SD-X4 (0 = limpio)",
        ),
    ) -> List[CogPath]:

        # 1️⃣ Imagen base SD-XL
        base = self.pipe_base(
            prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]

        if upscale_mode == "none":
            out = Path("output_base.png"); base.save(out); return [CogPath(out)]

        # 2️⃣ Latent ×2 (opcional)
        img_2x = base
        if upscale_mode in {"latent_x2", "combo_x2_then_x4"}:
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            img_2x = self.pipe_latent2x(
                prompt=prompt,
                image=base,
                num_inference_steps=up_steps,
                guidance_scale=guidance,
               # noise_level=latent_noise,
            ).images[0]
            if upscale_mode == "latent_x2":
                out = Path("output_2x.png"); img_2x.save(out); return [CogPath(out)]

        # 3️⃣ SD-X4 (opcional)
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

        # safety resize → máx 1024 px de input
        if max(img_2x.size) > MAX_X4_INPUT:
            w, h = img_2x.size
            scale = MAX_X4_INPUT / max(w, h)
            img_2x = img_2x.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS,
            )

        img_4x = self.pipe_x4(
            prompt=prompt,
            image=img_2x,
            num_inference_steps=up_steps,
            guidance_scale=guidance,
            noise_level=x4_noise,
        ).images[0]

        out = Path("output_4x.png"); img_4x.save(out)
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        return [CogPath(out)]

