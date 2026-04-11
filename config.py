from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/electro"
    port: int = 8001
    host: str = "0.0.0.0"
    model_path: str = "./models/recommendation_model.pkl"
    min_interactions: int = 5
    top_k_default: int = 12

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
MODEL_PATH = Path(settings.model_path)
