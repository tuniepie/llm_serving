import time
from typing import Optional, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dependencies import ml_models, generate_response
from configs.configs import settings

# Initialize router for chat-related endpoints
router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

# Define request and response models
class ChatMessage(BaseModel):
    """
    Request model for chat messages.
    
    Attributes:
        role (str): The role of the message sender (e.g., "user" or "assistant").
        content (str): The content of the message.
    """
    role: Optional[str]
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completions.
    
    Attributes:
        model (str): Model name.
        messages (List[ChatMessage]): List of chat messages.
        max_tokens (Optional[int]): Maximum number of tokens for the response.
        temperature (Optional[float]): Sampling temperature.
        stream (Optional[bool]): Whether to stream the response token by token.
    """
    model: str = settings.MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = settings.MAX_NEW_TOKENS
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    """
    Response model for chat completions.
    
    Attributes:
        id (str): Unique identifier for the response.
        object (str): Type of object (e.g., "chat.completion").
        created (float): Timestamp of response creation.
        model (str): Model name used for the response.
        choices (List): List of generated choices.
    """
    id: str
    object: str
    created: float
    model: str = settings.MODEL_NAME
    choices: List


@router.get("/")
async def root():
    """
    Root endpoint to check if the service is running.
    
    Returns:
        dict: A simple "Hello, World" message.
    """
    if "models" not in ml_models or "tokenizers" not in ml_models:
        llm_service_information = {
            "status": "not started",
            "message": "You need to start the LLM first."
        }
    else:
        llm_service_information = {
            "status": "running",
            "llm_serving_url": settings.LLM_SERVING_URL,
            "model_name": settings.MODEL_NAME,
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "device": settings.DEVICE,
            "do_sample": settings.DO_SAMPLE,
            "skip_special_tokens": settings.SKIP_SPECIAL_TOKENS,
        }

    return llm_service_information


@router.post("/completions")
async def chat(request: ChatCompletionRequest):
    """
    Endpoint to handle chat messages with OpenAI-compatible response format.
    
    Args:
        request (ChatCompletionRequest): The user's chat message and configuration.
    
    Returns:
        dict or StreamingResponse: The generated response, either fully or token-by-token.
    """
    # Combine all user messages into a single prompt
    prompt = request.messages
    print('Request model:', request.model)
    print("[INFO]:", prompt)

    if request.stream:
        # Stream response token by token
        return StreamingResponse(
            generate_response(prompt, model_name=request.model, stream=True), media_type="application/x-ndjson"
        )

    # Generate full response
    generated_text = generate_response(prompt, model_name=request.model)
    print("[INFO]:", generated_text)
    return {
        "id": "completion-id",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": generated_text}
        }]
    }
