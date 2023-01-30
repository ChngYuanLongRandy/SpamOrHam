from pydantic import BaseSettings
from typing import List
import os

class Settings (BaseSettings):
    API_NAME = "spam_or_ham"
    API_STR: str = os.environ.get("API_STR", "/api/v1")
    SCAMY_WORDS : List[str] = ['call','text','won','now','free']
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "api/api/v1/logreg_model")
    DATA_PATH: str = os.environ.get("MODEL_PATH", "api/api/v1/spam.csv")
    TEST_SIZE: float = os.environ.get("TEST_SIZE", 0.10)
    SEED: int = os.environ.get("SEED", 42)
    STRATIFY: bool = os.environ.get("STRATIFY", True)

SETTINGS = Settings()