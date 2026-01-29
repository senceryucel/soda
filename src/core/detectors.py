from io import BytesIO
from src.core.base import BaseDetector
import sys

class ApiDetector(BaseDetector):
    def __init__(self, model_id, box_threshold, text_threshold):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, image, text_prompt: str) -> dict:
        classes = [c.strip() for c in text_prompt.split(".") if c.strip()]
        
        if len(classes) > 1:
            print("\nERROR: api mode does not support multi-class detection")
            print(f"you requested {len(classes)} classes: {classes}")
            print("please use local mode for multi-class, or stick to a single class in api mode.")
            sys.exit(0)
            
        single_class_name = classes[0]

        buf = BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        import replicate
        from replicate.exceptions import ReplicateError
        
        succ = False
        retry = 0
        max_retries = 3
        wait_sec = 4
        output = None
        while not succ:
            try:
                output = replicate.run(
                    self.model_id,
                    input={
                        "image": buf,
                        "query": text_prompt,
                        "box_threshold": self.box_threshold,
                        "text_threshold": self.text_threshold
                    }
                )
                succ = True

            except ReplicateError as e:
                print(f"got rate limited by replicate. waiting for {wait_sec} seconds before retrying...")
                import time
                time.sleep(wait_sec)
                retry += 1
                if retry > max_retries:
                    print(f"couldn't handle replicate exception after {max_retries} retries. error: {e}")
                    sys.exit(0)
        
        boxes, scores, labels = [], [], []
        detections = output.get('detections', []) if isinstance(output, dict) else output
        
        for item in detections:
            if isinstance(item, str): 
                continue

            boxes.append(item['bbox'])
            scores.append(item.get('confidence') or item.get('score'))
            labels.append(single_class_name)
            
        return {"boxes": boxes, "scores": scores, "labels": labels}
    

class LocalDetector(BaseDetector):
    def __init__(self, model_id, device, box_threshold, text_threshold):
        print(f"loading local detection model: {model_id}")

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True)
        except (OSError, ValueError):
            print("detector model not found locally. downloading from hub...")
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=False)

    def detect(self, image, text_prompt: str) -> dict:
        import torch
        clean_prompt = text_prompt.strip()
        if clean_prompt.endswith("."):
            clean_prompt = clean_prompt[:-1].strip()
            
        self.model.to(self.device)
        inputs = self.processor(images=image, text=clean_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes
        )[0]

        final_labels = self._label_decoding(outputs, inputs, self.box_threshold)

        self.model.to("cpu")
        torch.cuda.empty_cache()
        
        return {
            "boxes": results["boxes"].cpu().numpy().tolist(),
            "scores": results["scores"].cpu().numpy().tolist(),
            "labels": final_labels
        }

    def _label_decoding(self, outputs, inputs, threshold):
        logits = outputs.logits.sigmoid()[0]
        
        # values: confidence scores, indices: token indices in input_ids
        max_scores, token_indices = logits.max(dim=1) 
        
        keep = max_scores > threshold
        keep_token_indices = token_indices[keep]
        
        decoded_labels = []
        input_ids = inputs.input_ids[0]
        
        for token_idx in keep_token_indices:
            if token_idx >= len(input_ids):
                decoded_labels.append("unknown")
                continue
                
            predicted_token_id = input_ids[token_idx]
            word = self.processor.tokenizer.decode([predicted_token_id], skip_special_tokens=True)
            clean_word = word.strip().replace(".", "")

            if not clean_word:
                decoded_labels.append("unknown")
            else:
                decoded_labels.append(clean_word)

        return decoded_labels