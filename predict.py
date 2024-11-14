# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import base64
from io import BytesIO
from cog import BasePredictor, Input, Path, BaseModel
import torch
import os
import subprocess
import time
import numpy as np
from PIL import Image
from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
    CLIPImageProcessor,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from openai import OpenAI

FALCON_MODEL_NAME = "Falconsai/nsfw_image_detection"
FALCON_MODEL_CACHE = "model-cache"
FALCON_MODEL_URL = (
    "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"
)

COMPVIS_MODEL_CACHE = "safety-cache"
COMPVIS_MODEL_FEATURE_EXTRACTOR = "feature-extractor"
COMPVIS_MODEL_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def download_weights(url, dest):
    if not os.path.exists(dest):
        start = time.time()
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
        print(f"downloading took: {time.time() - start:.2f} seconds")
    else:
        print(f"Skipping download, {dest} already exists")


def run_compvis_safety_checker(self, image):
    start_time = time.time()
    safety_checker_input = self.compvis_feature_extractor(
        image, return_tensors="pt"
    ).to("cuda")
    np_image = np.array(image)
    _, has_nsfw_concept = self.compvis_safety_checker(
        images=np_image,
        clip_input=safety_checker_input.pixel_values.to(torch.float16),
    )
    end_time = time.time()
    print(f"CompVis safety check took {end_time - start_time:.2f} seconds")
    return not has_nsfw_concept[0], end_time - start_time


def run_falcon_safety_checker(self, image):
    start_time = time.time()
    with torch.no_grad():
        inputs = self.falcon_processor(images=image, return_tensors="pt")
        outputs = self.falcon_model(**inputs)
        logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        result = self.falcon_model.config.id2label[predicted_label]
    end_time = time.time()
    print(f"Falcon safety check took {end_time - start_time:.2f} seconds")

    is_safe = result == "normal"
    return is_safe, end_time - start_time

def run_openai_safety_checker(self, image):
    start_time = time.time()
    # Convert PIL Image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    response = self.openai_client.moderations.create(
        model="omni-moderation-latest",
        input=[{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_str}"
            }
        }]
    )
    end_time = time.time()
    print(f"OpenAI safety check took {end_time - start_time:.2f} seconds")
    return response.results[0].flagged, end_time - start_time, response.results[0].categories, response.results[0].category_scores


class SafetyCheckResult(BaseModel):
    falcon_is_safe: bool
    falcon_time_taken: float
    compvis_is_safe: bool
    compvis_time_taken: float
    openai_is_safe: bool
    openai_categories: list[str]
    openai_scores: list[float]
    openai_time_taken: float


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        download_weights(FALCON_MODEL_URL, FALCON_MODEL_CACHE)
        download_weights(COMPVIS_MODEL_URL, COMPVIS_MODEL_CACHE)

        self.falcon_model = AutoModelForImageClassification.from_pretrained(
            FALCON_MODEL_NAME,
            cache_dir=FALCON_MODEL_CACHE,
        )
        self.falcon_processor = ViTImageProcessor.from_pretrained(FALCON_MODEL_NAME)

        self.compvis_safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            COMPVIS_MODEL_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.compvis_feature_extractor = CLIPImageProcessor.from_pretrained(
            COMPVIS_MODEL_FEATURE_EXTRACTOR
        )

        self.openai_client = OpenAI()

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> SafetyCheckResult:
        """Run a single prediction on the model"""
        img = Image.open(image)

        falcon_output, falcon_time = run_falcon_safety_checker(self, img)
        compvis_output, compvis_time = run_compvis_safety_checker(self, img)
        openai_output, openai_time, openai_categories, openai_scores = run_openai_safety_checker(self, img)
        print("Falcon output: ", falcon_output)
        print("Compvis output: ", compvis_output)
        print("OpenAI output: ", openai_output)

        return SafetyCheckResult(
            falcon_is_safe=falcon_output,
            falcon_time_taken=falcon_time,
            compvis_is_safe=compvis_output,
            compvis_time_taken=compvis_time,
            openai_is_safe=openai_output,
            openai_time_taken=openai_time,
            openai_categories=openai_categories,
            openai_scores=openai_scores,
        )
