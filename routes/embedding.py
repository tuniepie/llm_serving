from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..dependencies import embed_models, embed_texts
from ..configs.configs import Configs_Settings

# Initialize router for embedding-related endpoints
router = APIRouter(
    prefix="/embeddings",
    tags=["embedding"]
)

class EmbeddingObject(BaseModel):
    """
    Embedding object for returning embedding

    Attributes:
        object (str): The object type, which is always "embedding".
        embedding (List[float]): The embedding vector, which is a list of floats.
        index (int): The index of the embedding in the list of embeddings.
    """
    object: str = "embedding",
    embedding: List[float]
    index: int


class EmbedRequest(BaseModel):
    """
    Request embed for embedding texts.
    
    Attributes:
        input (List[str]): list of prompts from user.
        model (str): Embed model name.
        encoding_format (str): The format to return the embeddings in. Can be either `float` or `base64`.
        dimensions (int): The number of dimensions the resulting output embeddings should have.
        user (str): A unique identifier representing your end-user.
    """
    input: List[str] = [""]
    model: str = Configs_Settings.EMB_MODEL_NAME
    encoding_format: str = "float"
    dimensions: int = "512"
    user: str = "007"


class EmbedResponse(BaseModel):
    """
    Response embed for embedding text.
    
    Attributes:
        object (str): Type of object (e.g., "chat.completion").
        data: list of embedding object.
        model (str): embed model name used for embedding.
    """
    object: str = "list"
    data: List[EmbeddingObject]
    model: str = Configs_Settings.EMB_MODEL_NAME



@router.get("/")
async def root():
    """
    Root endpoint to check if the service is running.
    
    Returns:
        dict: embed service information
    """
    if "models" not in embed_models or "tokenizers" not in embed_models:
        embed_service_information = {
            "status": "not started",
            "message": "You need to start the embed model first."
        }
    else:
        embed_service_information = {
            "status": "running",
            "embedding_serving_url": Configs_Settings.EMB_SERVING_URL,
            "model_name": Configs_Settings.EMB_MODEL_NAME,
            "chunk_size": Configs_Settings.CHUNK_SIZE,
            "embedding_dim": Configs_Settings.EMBEDDING_DIM,
        }

    return embed_service_information


@router.post("/")
async def get_embeddings(request: EmbedRequest):
    """
    API endpoint to compute embeddings for the input texts.

    Args:
        request (EmbedRequest): Input data containing a list of strings to embed.

    Returns:
        EmbedResponse: Response containing a list of embeddings.

    Raises:
        HTTPException: If the input is invalid or if an error occurs during processing.
    """
    # Validate input: Ensure inputs are not empty or contain only whitespace
    if not request.input or any(not text.strip() for text in request.input):
        raise HTTPException(status_code=400, detail="Input texts must not be empty.")
    
    # Generate embeddings for the input texts
    embeddings = embed_texts(texts=request.input, model_name=request.model)
    data = [EmbeddingObject(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
    embeddings_response = EmbedResponse(data=data, model=request.model)
    return embeddings_response
