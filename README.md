# AI-Powered Video Marketing

AI-powered pipeline to create marketing video from product image.

## 1. Tutorial

- Recommended testing environment: **Google Colab** (since the pipeline consumes lots of GPU)
  - Enhance image: T4 GPU
  - Generate scene: T4 GPU
  - Generate video: A100 GPU
- Follow the executed workflow in [pipeline.ipynb](pipeline.ipynb)

## 2. Workflow

### 2.1 Enhance image

- Tech stacks:
  - GAN-based Super-Resolution: [fal/AuraSR-v2](https://huggingface.co/fal/AuraSR-v2) (variation of [GigaGAN](https://mingukkang.github.io/GigaGAN/))

- Sample:

| Input image | Enhanced image |
|:-----------:|:--------------:|
| ![](assets/0_sample_input_1024.png) | ![](assets/1_enhanced_image.png) |

> Diff: ![](assets/1_enhanced_diff.png)

### 2.2 Generate scene

- Tech stacks:
  - ZoeDetector: [lllyasviel/Annotators](lllyasviel/Annotators)
  - StableDiffusionXL-VAE: [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
  - ControlNet:
    - [destitech/controlnet-inpaint-dreamer-sdxl](https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl)
    - [diffusers/controlnet-zoe-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0)
  - StableDiffusionXLControlNetPipeline: [SG161222/RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)
  - StableDiffusionXLInpaintPipeline: [OzzyGT/RealVisXL_V4.0_inpainting](https://huggingface.co/OzzyGT/RealVisXL_V4.0_inpainting)

- Sample: ![](assets/2_generated_scene.png)

### 2.3 Generate video

- Tech stacks:
  - Stable Video Diffusion (SVD): [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)

- Sample:

## References

1. M. Kang *et al.*, “Scaling up gans for text-to-image synthesis,” 2023. [Online]. Available: https://arxiv.org/abs/2303.05511

2. S. F. Bhat *et al.*, “Zoedepth: Zero-shot transfer by combining relative and metric depth,” 2023. [Online]. Available: https://arxiv.org/abs/2302.12288

3. R. Rombach *et al.*, “High-resolution image synthesis with latent diffusion models,” 2022. [Online]. Available: https://arxiv.org/abs/2112.10752

4. Stability.AI, “Stable video diffusion: Scaling latent video diffusion models to large datasets,” 2023. [Online]. Available: https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets
