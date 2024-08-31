import os
import sys
import math
from typing import Optional

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")), "generative_models"))

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor

from modules.base import BaseModule

class VideoGenerator(BaseModule):
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        
        self.load_config()
        self.load_device()
        self.get_resources()

    def load_config(self) -> None:
        print("> LOAD CONFIG...")
        config = self.load_yaml(yaml_path=self.config_path)
        self.set_attributes(**config)
        
    def get_resources(self) -> None:
        local_dir = "checkpoints"
        local_file_path = os.path.join(local_dir, self.filename)

        if not os.path.exists(local_file_path):
            hf_hub_download(repo_id=self.repo_id, filename=self.filename, local_dir=local_dir)
            print("File downloaded.")
        else:
            print("File already exists. No need to download.")


    def generate_video(self,
        image: Image.Image,
        num_frames: Optional[int] = None,
        num_steps: Optional[int] = None,
        fps_id: int = 6,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        seed: int = 42,
        decoding_t: int = 7,
        verbose: Optional[bool] = False,
    ):
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        model_config = f"scripts/sampling/configs/{self.version}.yaml"

        self.model, filter = self.load_model(
            model_config,
            num_frames,
            num_steps,
            verbose,
        )
        self.set_seed(seed=seed)

        input_image = image.convert("RGB")
        w, h = image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(self.device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in self.version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in self.version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

        with torch.no_grad():
            with torch.autocast(self.device):
                batch, batch_uc = self.get_batch(
                    __class__.get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames
                )
                c, uc = self.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=self.device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(self.device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return self.model.denoiser(
                        self.model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = self.model.sampler(denoiser, randn, cond=c, uc=uc)
                self.model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = self.model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                samples = embed_watermark(samples)
                samples = filter(samples)
                video = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                
        self.clear_memory()
                    
        return video

    @staticmethod
    def get_unique_embedder_keys_from_conditioner(conditioner) -> list:
        return list(set([x.input_key for x in conditioner.embedders]))

    def get_batch(self,
        keys: list,
        value_dict: dict,
        N: list, T: int
    ) -> tuple:
        batch = {}
        batch_uc = {}

        for key in keys:
            if key == "fps_id":
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]])
                    .to(self.device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .to(self.device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to(self.device),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames" or key == "cond_frames_without_noise":
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            elif key == "polars_rad" or key == "azimuths_rad":
                batch[key] = torch.tensor(value_dict[key]).to(self.device).repeat(N[0])
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def load_model(self,
        config: str,
        num_frames: int,
        num_steps: int,
        verbose: bool = False,
    ) -> tuple:
        config = OmegaConf.load(config)
        if self.device == "cuda":
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = self.device
        
        config.model.params.sampler_config.params.verbose = verbose
        config.model.params.sampler_config.params.num_steps = num_steps
        config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )
        if self.device == "cuda":
            with torch.device(self.device):
                model = instantiate_from_config(config.model).to(self.device).eval()
        else:
            model = instantiate_from_config(config.model).to(self.device).eval()

        filter = DeepFloydDataFiltering(verbose=False, device=self.device)
        return model, filter
