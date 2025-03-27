import dotenv

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()

class Configs_Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra="ignore")

    # LLM 
    LLM_SERVING_URL: str = Field(default="http://localhost:8001")
    MAX_NEW_TOKENS: int = Field(default=512)
    MODEL_NAME: str = Field(default="bigscience/bloomz-1b1")
    # Qwen/Qwen2.5-3B-Instruct bigscience/bloomz-1b1
    DEVICE: str = Field(default="cuda:1")  # Change as needed
    DO_SAMPLE: bool = Field(default=True)
    SKIP_SPECIAL_TOKENS: bool = Field(default=True)

    # Embedding
    EMB_SERVING_URL: str = Field(default="http://localhost:8001")
    EMB_MODEL_NAME: str = Field(default="BAAI/llm-embedder")
    CHUNK_SIZE: int = Field(default=256)
    EMBEDDING_DIM: int = Field(default=768)

    HUGGINGFACE_KEY: str
    


settings = Configs_Settings()
