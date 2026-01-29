import os
import cv2
import random
import gc
import torch
import numpy as np

from src.utils.postprocessing import nms
from src.utils.prompting import process_wildcards
from src.utils.formatting import LabelFormatter
from src.core.base import BaseGenerator, BaseDetector

class DatasetPipeline:
    def __init__(self, generator: BaseGenerator, detector: BaseDetector, config: dict):
        self.generator = generator
        self.detector = detector
        
        self.output_dir = config["project"]["output_dir"]
        self.output_format = config["output"]["format"]
        self.save_empty = config["system"].get("save_empty_images", False)
        self.class_map = config["detection"].get("class_map", {})
        
        self.paths = {
            "images": os.path.join(self.output_dir, "images"),
            "debug": os.path.join(self.output_dir, "debug")
        }
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def run(self, gen_prompt: str, det_prompt: str, count: int):
        print(f"\nstarting pipeline. target: {count} samples. format: {self.output_format}")
        
        saved_count = 0
        
        for i in range(count):
            gc.collect()
            torch.cuda.empty_cache()
            
            current_seed = random.randint(0, 2**32 - 1)
            current_prompt = process_wildcards(gen_prompt)
            print(f"[{i+1}/{count}] generating... seed: {current_seed}")
            print(f"prompt: {current_prompt}")
            
            image = self.generator.generate(current_prompt, seed=current_seed)
            w, h = image.size
            
            results = self.detector.detect(image, det_prompt)
            
            keep_indices = nms(results["boxes"], results["scores"])
            final_boxes = [results["boxes"][k] for k in keep_indices]
            final_scores = [results["scores"][k] for k in keep_indices]
            final_labels = [results["labels"][k] for k in keep_indices]
            
            if not self.save_empty and len(final_boxes) == 0:
                print(f"samle {i} skipped: no objects detected.")
                continue
            
            filename = f"sample_{i:04d}_{current_seed}"
            
            image.save(os.path.join(self.paths["images"], f"{filename}.jpg"))
            
            LabelFormatter.save(
                self.output_format, 
                final_boxes, 
                final_labels, 
                filename, 
                self.output_dir, 
                w, h,
                class_map=self.class_map
            )
            
            self._save_debug(image, final_boxes, final_scores, final_labels, filename)
            saved_count += 1
            
        print(f"pipeline finished. {saved_count}/{count} images saved to {self.output_dir}")

    def _save_debug(self, image, boxes, scores, labels, filename):
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{label} {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.paths["debug"], f"{filename}_debug.jpg"), cv_img)