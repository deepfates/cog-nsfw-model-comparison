# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import os
import subprocess
import time
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

MODEL_NAME = "Falconsai/nsfw_image_detection"
MODEL_CACHE = "model-cache"
MODEL_URL = (
    "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"
)

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print(f"downloading took: {time.time() - start:.2f} seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        download_weights(MODEL_URL, MODEL_CACHE)
        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
        )
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> str:
        """Run a single prediction on the model"""
        img = Image.open(image)

        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_label = logits.argmax(-1).item()
        output = self.model.config.id2label[predicted_label]
        return output
