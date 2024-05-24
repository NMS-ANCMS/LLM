from typing import Any
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import tensorflow as tf
import ollama 

GENERATIVE_AI_MODEL_REPO = "TheBloke/Llama-2-7B-GGUF"
GENERATIVE_AI_MODEL_FILE = "llama-2-7b.Q4_K_M.gguf"

model_path = hf_hub_download(
    repo_id=GENERATIVE_AI_MODEL_REPO,
    filename=GENERATIVE_AI_MODEL_FILE
)

# llama2_model = Llama(
#     model_path=model_path,
#     n_gpu_layers=64,
#     n_ctx=2000,
#     chat_format="llama2"
# )
#
# llama2_model.create_chat_completion(
#     messages=[
#         {"role": "system", "content": "You are an Network assistant who perfectly troubleshoots failures"},
#         {
#             "role": "user",
#             "content": "what is the first step to troublshoot cisco interface failure?"
#         }
#     ]
# )

app = FastAPI()


class TextInput(BaseModel):
    inputs: str
    # parameters: dict[str, Any] | None
    user_id: str


@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }


@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        print(type(data))
        print(data)
        # params = data.parameters or {}
        response = ollama.chat(
            model='llama2',
            messages=[{'role': 'user', 'content': data.inputs}],
            stream=True,
        )
        # response = llama2_model(prompt=data.inputs, **params)
        model_out = response['choices'][0]['text']
        return {"generated_text": model_out}
    except Exception as e:
        print(type(data))
        print(data)
        raise HTTPException(status_code=500, detail=len(str(e)))
