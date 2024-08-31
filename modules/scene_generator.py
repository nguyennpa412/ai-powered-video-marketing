import torch

from PIL import Image, ImageOps
from modules.base import BaseModule
from controlnet_aux import ZoeDetector

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

TORCH_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bfp16": torch.bfloat16,
}

class SceneGenerator(BaseModule):
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        
        self.load_config()
        self.load_device()

    def load_config(self) -> None:
        print("> LOAD CONFIG...")
        config = self.load_yaml(yaml_path=self.config_path)
        self.set_attributes(**config)
        self.torch_dtype = TORCH_DTYPE[self.torch_dtype]
        
    def load_zoe_model(self) -> None:
        print("> LOAD ZOE MODEL...")
        self.model["zoe"] = ZoeDetector.from_pretrained(self.zoe_path).to(self.device)
    
    def load_inpaint_model(self) -> None:
        print(">> FREE MEMORY...")
        self.model["zoe"] = None
        self.clear_cache()
        
        print("> LOAD INPAINT MODEL...")        
        self.model["vae"] = AutoencoderKL.from_pretrained(self.vae_path, torch_dtype=self.torch_dtype).to(self.device)
        
        self.model["controlnets"] = []
        for cn in self.controlnets.keys():
            self.model["controlnets"].append(
                ControlNetModel.from_pretrained(torch_dtype=self.torch_dtype, **self.controlnets[cn])
            )
        
        self.model["sdxlcn"] = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.sdxlcn_path,
            torch_dtype=self.torch_dtype,
            variant=self.variant,
            controlnet=self.model["controlnets"],
            vae=self.model["vae"]
        ).to(self.device)
        
    def load_outpaint_model(self) -> None:
        print(">> FREE MEMORY...")
        self.model["controlnets"] = None
        self.model["sdxlcn"] = None
        self.clear_cache()
        
        print("> LOAD OUTPAINT MODEL...")
        self.model["sdxlip"] = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.sdxlip_path,
            torch_dtype=self.torch_dtype,
            variant=self.variant,
            vae=self.model["vae"]
        ).to(self.device)
        
    def get_generator(self, seed: int = None) -> None:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        
    def get_paste_position(self) -> None:
        self.paste_x = (self.resized_width - self.resized_img.width) // 2
        self.paste_y = (self.resized_height - self.resized_img.height) // 2
        
    def paste_original(self, bg: Image.Image) -> None:
        bg.paste(
            self.resized_img,
            (self.paste_x, self.paste_y),
            self.resized_img
        )
        
    def zoe_detect(self,
        detect_resolution: int = 512,
        image_resolution: int = 1024
    ) -> None:
        self.zoe_image = self.model["zoe"](
            self.white_bg_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution
        )

    def inpaint(self) -> None:
        self.inpainted_image = self.model["sdxlcn"](
            self.prompt,
            negative_prompt=self.negative_prompt,
            image=[self.white_bg_image, self.zoe_image],
            guidance_scale=self.sdxlcn_guidance_scale,
            num_inference_steps=self.sdxlcn_num_inference_steps,
            generator=self.generator,
            controlnet_conditioning_scale=self.sdxlcn_controlnet_conditioning_scale,
            control_guidance_end=self.sdxlcn_control_guidance_end,
        ).images[0]
        self.paste_original(bg=self.inpainted_image)
        
    def create_outpaint_mask(self) -> None:
        mask = Image.new("L", self.inpainted_image.size)
        mask.paste(self.resized_img.split()[3], (self.paste_x, self.paste_y))
        mask = ImageOps.invert(mask)
        final_mask = mask.point(lambda p: p > 128 and 255)
        self.outpaint_mask_blurred = self.model["sdxlip"].mask_processor.blur(final_mask, blur_factor=self.sdxlip_mask_blur_factor)
        
    def outpaint(self) -> None:
        adjusted_prompt = f"High quality photo of {self.prompt}, highly detailed, professional"
        self.outpainted_image = self.model["sdxlip"](
            adjusted_prompt,
            negative_prompt=self.negative_prompt,
            image=self.inpainted_image,
            mask_image=self.outpaint_mask_blurred,
            guidance_scale=self.sdxlip_guidance_scale,
            strength=self.sdxlip_strength,
            num_inference_steps=self.sdxlip_num_inference_steps,
            generator=self.generator,
        ).images[0]
        self.paste_original(bg=self.outpainted_image)
        
    def create_scene(self,
        image_path: str,
        prompt: str,
        is_url: bool = False,
        negative_prompt: str = "",
        resized_size: tuple[int] = (1024,1024),
        seed: int = None,
        zoe_detect_resolution: int = 512,
        zoe_image_resolution: int = 1024
    ) -> Image.Image:        
        if is_url:
            self.original_image = self.load_image_from_url(url=image_path)
        else:
            self.original_image = Image.open(image_path)

        self.resized_img, self.white_bg_image = self.scale_and_paste(
            original_image=self.original_image,
            resized_size=resized_size,
            background_color="white"
        )
        self.get_generator(seed=seed)
        self.resized_width = resized_size[0]
        self.resized_height = resized_size[1]
        self.get_paste_position()
        
        self.model = {}
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        
        with torch.inference_mode():
            self.load_zoe_model()
            self.zoe_detect(detect_resolution=zoe_detect_resolution, image_resolution=zoe_image_resolution)
            self.load_inpaint_model()
            self.inpaint()
            self.load_outpaint_model()
            self.create_outpaint_mask()
            self.outpaint()
        
        self.clear_memory()
        
        return self.outpainted_image
        