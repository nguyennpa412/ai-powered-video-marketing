import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from aura_sr import AuraSR
from modules.base import BaseModule

class Enhancer(BaseModule):
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        
        self.load_config()
        self.load_device()
        self.load_model()

    def load_config(self) -> None:
        print("> LOAD CONFIG...")
        config = self.load_yaml(yaml_path=self.config_path)
        self.set_attributes(**config)
    
    def load_model(self) -> None:
        print("> LOAD MODEL...")
        self.model = AuraSR.from_pretrained(self.model_path)

    def enhance(self,
        image_path: str,
        is_url: bool = False,
        resized_size: tuple[int] = (1024,1024)
    ) -> Image.Image:
        if is_url:
            self.input_image = self.load_image_from_url(url=image_path)
        else:
            self.input_image = Image.open(image_path)
            
        _, self.input_image = __class__.scale_and_paste(
            original_image=self.input_image,
            resized_size=resized_size,
        )
        img_3c = np.array(self.input_image)[:,:,:3]
        img_c4 = np.array(self.input_image)[:,:,3]
        
        with torch.inference_mode():
            self.enhanced_image = self.model.upscale_4x_overlapped(image=Image.fromarray(img_3c))
            
        _, self.enhanced_image = __class__.scale_and_paste(
            original_image=self.enhanced_image,
            resized_size=resized_size,
            is_transparent_output=False
        )
        self.enhanced_image = Image.fromarray(np.dstack((np.array(self.enhanced_image), img_c4)))
        
        return self.enhanced_image
    
    def plot(self):
        diff = 255 - cv2.absdiff(
            self.pil_to_cv2(pil_image=self.enhanced_image),
            self.pil_to_cv2(pil_image=self.input_image)
        )
        diff = Image.fromarray(diff)
        
        f, axes = plt.subplots(1,3, figsize=(20,20))
        axes[0].imshow(self.input_image)
        axes[0].set_title('input')
        axes[1].imshow(self.enhanced_image)
        axes[1].set_title('enhanced')
        axes[2].imshow(diff)
        axes[2].set_title('diff')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])