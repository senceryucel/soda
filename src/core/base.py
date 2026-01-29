from abc import ABC, abstractmethod
from PIL import Image

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, seed: int = None) -> Image.Image:
        pass

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image, text_prompt: str) -> dict:
        pass