# LLM


## Get Started
- Download **`llama-2`** model from [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)
- Put pdf files in **`data`** directory
- Run **`ingest.py`** to create vector database
- Run **`uvicorn app:app --reload`** to start chatbot
