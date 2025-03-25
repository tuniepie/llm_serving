from threading import Thread
from typing import Dict, List, Generator, Union, Optional

import json
import torch
from fastapi import HTTPException
from loguru import logger
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM, AutoModel
from queue import Queue
import uuid
import time

from configs.configs import settings

# ==========================
# Global ML Models, Embedder Registry
# ==========================
ml_models: Dict = {
    "models": {},
    "tokenizers": {},
}
embed_models: Dict = {
    "models": {},
    "tokenizers": {},
}

def load_model(model_name):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=settings.DEVICE,  # Adjust device as needed
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Register model and tokenizer in the global dictionary
    ml_models["models"][model_name] = model
    ml_models["tokenizers"][model_name] = tokenizer
    logger.info("LLM added to ml_models.")

def load_embedder(model_name):
    # Load the embedder and embed tokenizer
    embedder = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=settings.DEVICE
    )
    embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedder.eval()

    # Register embedder and embed tokenizer in the global dictionary
    embed_models["models"][model_name] = embedder
    embed_models["tokenizers"][model_name] = embed_tokenizer
    logger.info("Embedder added to embed_models.")

def check_model_name_format(model_name: str=settings.MODEL_NAME):
    if len(model_name.split("/")) < 2:
        return False
    return True


class CustomStreamer(TextIteratorStreamer):
    """
    A custom streamer class for handling tokenized text generation in a streaming fashion.

    This class extends the `TextIteratorStreamer` and provides mechanisms to queue generated
    text, handle streaming termination, and generate completion chunks compatible with a OpenAI's API format.

    Attributes:
        text_queue (Queue): A thread-safe queue to hold chunks of generated text.
        stop_signal (str): A signal to indicate the end of the streaming process.
        timeout (Optional[float]): The maximum wait time (in seconds) for retrieving data from the queue.
        id (UUID): A unique identifier for the streamer instance.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used for decoding text.
        model_name (str): name of model used to embed text.
        skip_prompt (bool, optional): Whether to skip the prompt during tokenization. Defaults to False.
        timeout (Optional[float], optional): The timeout duration for the queue operations. Defaults to None.
        **decode_kwargs: Additional keyword arguments for decoding the text.

    Methods:
        on_finalized_text(text: str, stream_end: bool = False):
            Processes finalized text by putting it in the queue. Adds a stop signal to the queue if the stream ends.
        
        __iter__():
            Returns the iterator object (self).
        
        __next__():
            Retrieves the next item from the text queue. Raises StopIteration if the stop signal is encountered.
    """
    def __init__(
        self, tokenizer: "AutoTokenizer", model_name=settings.MODEL_NAME, skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = "data: [DONE]\n\n"
        self.timeout = timeout
        self.id = uuid.uuid4()
        self.model_name = model_name

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        completion_chunk_object = {
            "id": str(self.id),
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": text
                    },
                    "logprobs": None,
                    "finish_reason": None,
                }
            ]
        }
        
        self.text_queue.put(f"data: {json.dumps(completion_chunk_object)}\n\n", timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


def generate_response(prompt: list, model_name: str = settings.MODEL_NAME, stream: bool = False) -> Union[str, Generator[str, None, None]]:
    """
    Generate a text response from the model, either as a complete string or streamed token-by-token.
    
    Args:
        prompt (str): Input prompt for text generation.
        model_name (str): name of model which is used to embed text.
        stream (bool): Whether to stream tokens as they are generated. Defaults to False.
    
    Returns:
        Union[str, Generator[str, None, None]]: Generated response text or a generator for streaming tokens.
    
    Raises:
        HTTPException: If the model or tokenizer is not initialized.
    """
    # Ensure model and tokenizer are initialized
    if "models" not in ml_models or "tokenizers" not in ml_models:
        logger.error("RAG service is not initialized.")
        raise HTTPException(status_code=500, detail="RAG service is not initialized.")

    if not check_model_name_format(model_name):
        logger.error(f"Model name format is invalid. Use default model: {settings.MODEL_NAME}")
        model_name = settings.MODEL_NAME

    if model_name not in ml_models["models"] or model_name not in ml_models["tokenizers"]:
        load_model(model_name)

    model = ml_models["models"][model_name]
    tokenizer = ml_models["tokenizers"][model_name]

    # Prepare input text with chat history and generation prompt
    input_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input for the model
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Configure generation parameters
    generation_kwargs = dict(
        model_inputs, 
        max_new_tokens=settings.MAX_NEW_TOKENS,
        do_sample=settings.DO_SAMPLE
    )

    if not stream:
        # Generate full response
        output = model.generate(**generation_kwargs)
        generated_text = tokenizer.decode(output[0], skip_prompt=True, skip_special_tokens=settings.SKIP_SPECIAL_TOKENS)
        return generated_text
    else:
        # Use TextIteratorStreamer for token-by-token generation
        streamer = CustomStreamer(tokenizer, model=model_name, skip_prompt=True, skip_special_tokens=settings.SKIP_SPECIAL_TOKENS)

        # Run generation in a separate thread
        generation_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

def embed_texts(texts: List[str], model_name: str=settings.EMB_MODEL_NAME) -> List[List[float]]:
    """
    Generate embeddings for a list of input texts.

    Args:
        texts (List[str]): List of input strings to embed.

    Returns:
        List[List[float]]: List of embeddings, where each embedding is represented as a list of float values.

    Raises:
        HTTPException: If an error occurs during embedding generation.
    """
    if "models" not in embed_models or "tokenizers" not in embed_models:
        logger.error("RAG service is not initialized.")
        raise HTTPException(status_code=500, detail="RAG service is not initialized.")

    if not check_model_name_format(model_name):
        logger.error(f"Model name format is invalid. Use default model: {settings.MODEL_NAME}")
        model_name = settings.MODEL_NAME

    if model_name not in embed_models["models"] or model_name not in embed_models["tokenizers"]:
        load_embedder(model_name)

    embed_model = embed_models["models"][model_name]
    embed_tokenizer = embed_models["tokenizers"][model_name]

    try:
        embeddings = []
        for text in texts:
            # Tokenize inputs
            input = embed_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(settings.DEVICE)
            
            # Compute embeddings
            with torch.no_grad():
                outputs = embed_model(**input)
                embedding = outputs.last_hidden_state[:, 0].tolist()[0]
                embeddings.append(embedding)
        
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
