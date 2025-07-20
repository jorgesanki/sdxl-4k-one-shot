# SDXL-4K-One-Shot 🚀

Texto → **4 K** en **1 sola llamada**  
Pipeline: SD-XL ➜ Latent ×2 ➜ SD-X4 (CUDA 12.8, VRAM ≤ 9 GB).

| input          | default | notas |
|----------------|---------|-------|
| prompt         | —       | texto principal |
| width/height   | 768     | 512-1024 |
| upscale_mode   | combo_x2_then_x4 | none · latent_x2 · sd_x4 · combo_x2_then_x4 |
| steps/up_steps | 24 / 12 | pasos por fase |
| guidance       | 5.0     | CFG scale |

### Ejemplos

```bash
# 1. Imagen 768 px
replicate run USER/sdxl-4k-one-shot \
  -i prompt="Epic sunrise over the Andes"

# 2. 4 K one-shot
replicate run USER/sdxl-4k-one-shot \
  -i prompt="Blue dragon over snowy peaks" \
  -i width=512 -i height=512
