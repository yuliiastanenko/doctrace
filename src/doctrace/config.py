from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    data_dir: Path = ROOT / "data"
    chunk_max_chars: int = 1500
    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


def collection_name(model: str, max_chars: int) -> str:
    return f"{model.split('/')[-1].lower()}_{max_chars}"


settings = Settings()
