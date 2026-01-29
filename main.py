import yaml
import os
import sys

from src.core.generators import LocalGenerator, ApiGenerator
from src.core.detectors import LocalDetector, ApiDetector
from src.core.pipeline import DatasetPipeline


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    mode = cfg["project"]["mode"]
    
    if mode == "api":
        token = os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            print("error: replicate api token is missing. please export it as 'REPLICATE_API_TOKEN'")
            sys.exit(1)

    print(f"initializing prompt-to-dataset in '{mode}' mode...")

    if mode == "api":
        gen = ApiGenerator(
            model_id=cfg["generation"]["api_model_id"],
            gen_params=cfg["generation"]["params"]
        )
        det = ApiDetector(
            model_id=cfg["detection"]["api_model_id"],
            box_threshold=cfg["detection"]["box_threshold"],
            text_threshold=cfg["detection"]["text_threshold"]
        )
    elif mode == "local":
        gen = LocalGenerator(
            model_id=cfg["generation"]["local_model_id"],
            device=cfg["project"]["device"],
            optimize_gpu=cfg["system"]["optimize_gpu"],
            gen_params=cfg["generation"]["params"]
        )
        det = LocalDetector(
            model_id=cfg["detection"]["local_model_id"],
            device=cfg["project"]["device"],
            box_threshold=cfg["detection"]["box_threshold"],
            text_threshold=cfg["detection"]["text_threshold"]
        )
    else:
        print(f"error: unknown mode '{mode}'")
        sys.exit(1)

    pipeline = DatasetPipeline(
        generator=gen,
        detector=det,
        config=cfg
    )

    pipeline.run(
        gen_prompt=cfg["generation"]["prompt"],
        det_prompt=cfg["detection"]["prompt"],
        count=cfg["generation"]["count"]
    )

if __name__ == "__main__":
    main()