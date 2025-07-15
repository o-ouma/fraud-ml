from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str
    PROJECT_NAME: str
    PROJECT_VERSION: str
    PROJECT_DESCRIPTION: str
    MODEL_PATH: Path

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_value: str) -> Optional[str | Path]:
            if field_name == "MODEL_PATH":
                return Path(raw_value)
            return raw_value

settings = Settings()
