from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    project_root: Path = Path(__file__).parents[2]
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


settings = Settings()
print(settings.project_root)
