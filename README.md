# Model Serving's API

This is the quick guide to using the APIs for serving models. 

## Usage

### How to run

To run the APIs seperately, go to the root directory: `llm_serving/` and run

```
uvicorn main:app --host 0.0.0.0 --port 8001
```

We've already started a model serving on the server `localhost:8001`. You can test APIs by using the example code below:

```python
import os
from openai import OpenAI

os.environ['NO_PROXY'] = "*"

client = OpenAI(
    api_key="OPENAI_API_KEY",  # This is the default and can be omitted
    base_url = "http://localhost:8001/"
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "tell me about Microsoft",
        }
    ],
    model="gpt-4o",
    stream=True,
)

for chunk in stream:
    # print(chunk)
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Functions Overview

#### Function: `lifespan` (async)

* **Description:** Lifespan context manager for FastAPI app lifecycle. Initializes the ML model, tokenizer, embedder and embed tokenizer when the app starts and cleans up resources when the app stops.
- **Parameters:**
    * `app` (FastAPI): FastAPI application instance.
- **Returns:** What the function returns.
    * None, keyword `yield` is used to maintain the FastAPI instance
- **Example:**
```python
app = FastAPI(
    title="Model Serving API",
    description="API for generating and streaming model outputs.",
    lifespan=lifespan
)
```

#### Function: `generate_response`

- **Description:** Generates a response based on the input prompt using a pre-trained language model.
- **Parameters:**
    - `prompt` (str): The input prompt for text generation.
    - `model_name` (str): name of model which is used to generate response.
    - `stream` (bool): stream a response or not.
- **Returns:** 
    - `str`: The generated response text.
- **Raises:**
    - `HTTPException`: If the model or tokenizer is not initialized.
- **Example:**
```python
generated_text = generate_response("Give me a news update on today's technology trends.", stream=False)
print(generated_text)
```

#### Function: `embed_texts`
- **Description:** Generates embeddings for a list of input texts by tokenizing the inputs, passing them through an embedding model, and processing the model's output to obtain the embeddings.
- **Parameters:**
    - `texts` (List[str]): A list of input strings to be embedded.
- **Returns:**
    - `List[List[float]]`: A list of embeddings, where each embedding is represented as a list of floating-point values.
- **Raises:**
    - `HTTPException`: Raised if:
        - The embedding service (embedder or embed_tokenizer) is not initialized.
        - An error occurs during embedding generation.

### API Endpoints

#### Root Endpoint: `/chat`
- **Description:** Simple endpoint to test if the service is working
- **Returns:**
    - If the service was started, returns a JSON object
    ```json
    {
        "status": "running",
        "llm_serving_url": "http://localhost:8001",
        "model_name": "bigscience/bloomz-1b1",
        "max_new_tokens": 1024,
        "device": "cuda:1",
        "do_sample": true,
        "system_prompt": "You are ...."
    }
    ```
    - Otherwise, returns
    ```json
    {
        "status": "not started",
        "message": "You need to start the service first."
    }
    ```

#### Generate Endpoint: `/chat/completions`

- **Description:** Accepts a prompt via POST and returns the complete generated response
- **Request body:** 
    - `ChatCompletionRequest`: A JSON body containing the input prompt. Example:
        ```json
        <!-- For Bloom model:  -->
        {
            "model": "bigscience/bloomz-1b1",
            "messages": [
                {
                "role": "string",
                "content": "Translate to English: Je t’aime."
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "stream": false
            }
        <!-- For orther model -->
        {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello! Talk about the News."
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "stream": false
        }
        
        ```
- **Returns:** A JSON body containing the output. Example:
    ```json
    {
        "id": "completion-id",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": generated_text}
        }]
    }
    ```

#### Generate Endpoint: `/embeddings`

- **Description:** Accepts a prompt via POST and returns the complete generated response
- **Request body:** 
    * `ChatCompletionRequest`: A JSON body containing the input prompt. Example:
        ```json
        {
            "model": "model name",
            "input": ["Hello, world!", "How are you?"],
            "encoding_format": "float"
        }
        ```
- **Returns:** A JSON body containing the output. Example:
    ```json
    {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    .... (1536 floats total for ada-002)
                    -0.0028842222,
                ],
                "index": 0
            }
        ],
        "model": "text-embedding-ada-002",
    }
    ```

### Class Definition
#### `ChatCompletionRequest(BaseModel)`
- **Description:** Input schema for the text generation prompt.
- **Attributes:**
    - `model` (str): model name.
    - `messages` (List[str]): list of messages.
    - `stream` (bool): streaming a response or not.

#### `ChatCompletionResponse(BaseModel)`
- **Description:** Input schema for the text generation prompt.
- **Attributes:**
    - `id` (int): message id.
    - `object` (str): unknown string.
    - `created` (Time.time): the time of created message.
    - `choices` (List[Dict]): list of messages.

#### `EmbeddingObject(BaseModel)`
- **Description:** Represents an individual embedding object containing metadata and the embedding vector.
- **Attributes**:
    - `object` (str): Always "embedding", indicating the type of object.
    - `embedding` (List[float]): The embedding vector, a list of floating-point values.
    - `index` (int): The position of the embedding within a list of embeddings.

#### `EmbedRequest(BaseModel)`
- **Description:** Used for requesting embeddings for a list of input texts.
- **Attributes:**
    - `input` (List[str]): A list of text prompts for which embeddings are requested. Defaults to [""].
    - `model` (str): The name of the embedding model. Defaults to "mock-embed-model".
    - `encoding_format` (str): Format of the embedding vector in the response. Possible values: "float" or "base64". Defaults to "float".
    - `dimensions` (int): The dimensionality of the embeddings. Defaults to 512.
    - `user` (str): A unique identifier representing the end-user. Defaults to "007".

#### `EmbedResponse(BaseModel)`
- **Description:** Used to encapsulate the response from the embedding service, containing metadata and embedding data.
- **Attributes**:
    - `object` (str): Type of object, defaults to "list".
    - `data` (List[EmbeddingObject]): A list of EmbeddingObject instances representing the embeddings.
    - `model` (str): The name of the embedding model used for generating embeddings. Defaults to "mock-embed-model".

#### `CustomStreamer(TextIteratorStreamer)`
- **Description:** The CustomStreamer class is designed to handle tokenized text generation in a streaming fashion. It extends the TextIteratorStreamer class and introduces additional functionality for queuing generated text, signaling stream termination, and formatting completion chunks compatible with OpenAI's API.
- **Attributes:**
    - `text_queue` (Queue): A thread-safe queue that holds generated text chunks.
    - `stop_signal` (str): A predefined signal (`data: [DONE]\n\n`) to indicate the end of the stream.
    - `timeout` (Optional[float]): The maximum wait time (in seconds) for retrieving items from the queue.
    - `id` (UUID): A unique identifier for the streamer instance.
    - `model_name` (str): name of model which is used to embed text.
