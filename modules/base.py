import gc
import yaml
import time
import torch
import random
import requests
import numpy as np

from PIL import Image
from io import BytesIO
from typing import Literal

class BaseModule():
    def set_attributes(self, **kwargs) -> None:
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

    def set_device(self, device: str = "cpu") -> None:
        self.device = device
        if self.device == "cpu":
            self.use_gpu = False
        else:
            self.use_gpu = True

    def load_device(self, cuda_only: bool = False) -> None:
        print("> LOAD DEVICE...")
        device = "cpu"
        if self.use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
                print(">> Using GPU...")
            else:
                print(">> No CUDA available, using CPU instead...")
        else:
            print(">> Using CPU...")

        assert not (cuda_only & (device == "cpu")), "GPU-only model!"

        self.set_device(device)

    def set_seed(self, seed: int = 42) -> None:
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

    def clear_memory(self) -> None:
        print("> CLEAR MEMORY...")
        self.model = None
        self.clear_cache()

    def clear_cache(self) -> None:
        time.sleep(1)
        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()
            
    @staticmethod
    def load_yaml(yaml_path: str) -> dict:
        with open(yaml_path, "r") as yamlfile:
            return(yaml.load(stream=yamlfile, Loader=yaml.FullLoader))
        
    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        response = requests.get(url)
        image_data = BytesIO(response.content)

        return Image.open(image_data)

    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        pil_image = pil_image.convert('RGB')
        cv2_image = np.array(pil_image)
        # Convert RGB to BGR
        cv2_image = cv2_image[:, :, ::-1].copy()

        return cv2_image
    
    @staticmethod
    def scale_and_paste(
        original_image: Image.Image,
        resized_size: tuple = (1024,1024),
        is_transparent_output: bool = True,
        background_color: Literal["white","black","transparent"] = "transparent"
    ) -> tuple:
        target_ratio = resized_size[1] / resized_size[0]
        im_ratio = original_image.height / original_image.width
        if target_ratio > im_ratio:
            # It must be fixed by width
            new_width = resized_size[0]
            new_height = round(new_width * im_ratio)
        else:
            # Fixed by height
            new_height = resized_size[1]
            new_width = round(new_height / im_ratio)

        resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
        if background_color == "transparent":
            background_color = (0,0,0,0)
        with_background = Image.new('RGBA', resized_size, background_color)
        x = (resized_size[0] - new_width) // 2
        y = (resized_size[1] - new_height) // 2
        
        if np.array(resized_original).shape[-1] == 4:
            with_background.paste(resized_original, (x, y), resized_original)
        else:
            with_background.paste(resized_original, (x, y))
            
        if not is_transparent_output:
            with_background = with_background.convert("RGB")
            
        return resized_original, with_background

    # @staticmethod
    # def scale_and_paste(
    #     original_image: Image.Image,
    #     resized_size: tuple = (1024,1024),
    #     is_transparent_output: bool = True,
    #     background_color: Literal["white","black","transparent"] = "transparent"
    # ) -> tuple:
    #     aspect_ratio = original_image.width / original_image.height

    #     if original_image.width > original_image.height:
    #         new_width = resized_size[0]
    #         new_height = round(new_width / aspect_ratio)
    #     else:
    #         new_height = resized_size[1]
    #         new_width = round(new_height * aspect_ratio)

    #     resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)

    #     if background_color == "transparent":
    #         background_color = (0,0,0,0)
    #     with_background = Image.new("RGBA", resized_size, background_color)
    #     x = (resized_size[0] - new_width) // 2
    #     y = (resized_size[1] - new_height) // 2
        
    #     if np.array(resized_original).shape[-1] == 4:
    #         with_background.paste(resized_original, (x, y), resized_original)
    #     else:
    #         with_background.paste(resized_original, (x, y))
            
    #     if not is_transparent_output:
    #         with_background = with_background.convert("RGB")

    #     return resized_original, with_background