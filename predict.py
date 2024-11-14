# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import base64
from io import BytesIO
from cog import BasePredictor, Input, Path, BaseModel

import time
from PIL import Image
from openai import OpenAI

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
    return response.results[0].flagged, end_time - start_time, response.results[0].categories, response.results[0].scores


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
        self.openai_client = OpenAI()

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> SafetyCheckResult:
        """Run a single prediction on the model"""
        img = Image.open(image)

        openai_output, openai_time, openai_categories, openai_scores = run_openai_safety_checker(self, img)
        print("OpenAI output: ", openai_output)

        return SafetyCheckResult(
            openai_is_safe=openai_output,
            openai_time_taken=openai_time,
            openai_categories=openai_categories,
            openai_scores=openai_scores,
        )
