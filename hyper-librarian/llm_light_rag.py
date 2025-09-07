from warm_retriever import WarmRetriever
from llm_client_llamacpp import LLMClient
from llm_light_rag import llm_light_answer

# Choose the 3B model for CPU speed
MODEL_PATH = "models/qwen2.5-3b-instruct-q5_k_m.gguf"

retr = WarmRetriever()  # already wired to DuckDB/HNSW
llm = LLMClient(model_path=MODEL_PATH, n_threads=8)  # adjust threads

question = "What drove revenue growth in 2023?"
result = llm_light_answer(retr, llm, question)

print(result)
