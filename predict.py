# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import base64
from io import BytesIO
from cog import BasePredictor, Input, Path, BaseModel, Secret
from typing import List

import time
from PIL import Image
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

def run_openai_safety_checker(self, image: Image.Image):
    start_time = time.time()
    try:
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
        logger.info(f"OpenAI safety check took {end_time - start_time:.2f} seconds")
        
        result = response.results[0]
        flagged = result.flagged
        
        # Convert Categories and CategoryScores to dictionaries using vars()
        categories_dict = vars(result.categories)
        categories = [category for category, is_flagged in categories_dict.items() if is_flagged]
        
        scores_dict = vars(result.category_scores)
        scores = list(scores_dict.values())
        
        return flagged, end_time - start_time, categories, scores
    except Exception as e:
        logger.error(f"An error occurred during safety checking: {e}")
        return False, 0.0, [], []


class SafetyCheckResult(BaseModel):
    openai_is_safe: bool
    openai_categories: List[str]
    openai_scores: List[float]
    openai_time_taken: float


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
    def predict(
        self,
        image: Path = Input(description="Input image"),
        api_key: Secret = Input(description="API key for OpenAI")
    ) -> SafetyCheckResult:
        """Run a single prediction on the model"""
        self.openai_client = OpenAI(api_key=api_key.get_secret_value())

        img = Image.open(image)

        openai_output, openai_time, openai_categories, openai_scores = run_openai_safety_checker(self, img)
        logger.info("OpenAI output: %s", openai_output)
        logger.info("OpenAI categories: %s", openai_categories)
        logger.info("OpenAI scores: %s", openai_scores)

        return SafetyCheckResult(
            openai_is_safe=not openai_output,  # Invert the output since OpenAI returns True for flagged content
            openai_time_taken=openai_time,
            openai_categories=openai_categories,
            openai_scores=openai_scores,
        )
