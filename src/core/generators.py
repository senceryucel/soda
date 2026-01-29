import requests
import random
from io import BytesIO
from PIL import Image

from src.core.base import BaseGenerator

class ApiGenerator(BaseGenerator):
    def __init__(self, model_id, gen_params=None):
        self.model_id = model_id
        self.params = gen_params or {}

    def generate(self, prompt: str, seed: int = None) -> Image.Image:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        # input schema of replicate
        input_args = {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "output_format": "jpg",
            "safety_filter_level": "block_medium_and_above",
            "seed": seed,
            "num_inference_steps": self.params.get("num_inference_steps"),
            "guidance_scale": self.params.get("guidance_scale")
        }

        import replicate
        from replicate.exceptions import ReplicateError
        succ = False
        retry = 0
        max_retries = 3
        wait_sec = 4
        output = None
        while not succ:
            try:
                output = replicate.run(self.model_id, input=input_args)
                succ = True
            except ReplicateError as e:
                print(f"got rate limited by replicate. waiting for {wait_sec} seconds before retrying...")
                import time
                time.sleep(wait_sec)
                retry += 1
                if retry > max_retries:
                    import sys
                    print(f"couldn't handle replicate exception. please check explicitly: {e}")
                    sys.exit(0)

        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

class LocalGenerator(BaseGenerator):
    def __init__(self, model_id, device="cuda", optimize_gpu=True, gen_params=None):
        print(f"loading local generation model: {model_id}")
        
        import torch
        from diffusers import FluxPipeline
        
        self.device = device
        self.params = gen_params or {}
        
        try:
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
        except (OSError, ValueError):
            print("generator model not found locally. downloading from hub...")
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=False
            )
        
        # gpu optimization
        if optimize_gpu:
            print("\nvram optimization enabled (sequential offload). expect much slower generation with much lower vram usage.\n")
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_attention_slicing()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
        else:
            self.pipe.to(device)
        
    def generate(self, prompt: str, seed: int = None) -> Image.Image:
        import torch
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        height = self.params.get("height")
        width = self.params.get("width")
        steps = self.params.get("num_inference_steps")
        guidance = self.params.get("guidance_scale")
        max_seq = self.params.get("max_sequence_length")
            
        image = self.pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=max_seq,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        
        return image