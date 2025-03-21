from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from loguru import logger

from .routes import chat, embedding
from .configs.configs import Configs_Settings
from .dependencies import ml_models, embed_models
from .dependencies import load_model, load_embedder

from huggingface_hub import login
login(token=Configs_Settings.HUGGINGFACE_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI app lifecycle.
    Initializes the ML model, tokenizer, embedder, and embed tokenizer
    when the app starts and cleans up resources when the app stops.

    Args:
        app (FastAPI): The FastAPI application instance.
    
    Yields:
        None: Execution will be paused to maintain lifespan context.
    """
    # Load the model and tokenizer
    load_model(Configs_Settings.MODEL_NAME)
    load_embedder(Configs_Settings.EMB_MODEL_NAME)
    
    # Maintain context
    yield

    # Clean up after app shutdown
    ml_models.clear()
    logger.info("ml_models cleared.")
    embed_models.clear()
    logger.info("embed_models cleared.")

# Initialize FastAPI app
app = FastAPI(
    title="Model Serving API",
    description="API for generating and streaming model outputs.",
    lifespan=lifespan
)

app.include_router(chat.router)
app.include_router(embedding.router)

@app.get("/")
async def root():
    return {"message": "Hello, World!"}



# ==========================
# Main Server Run
# ==========================
# if __name__ == "__main__":
#     # Start the FastAPI app using uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8001)
