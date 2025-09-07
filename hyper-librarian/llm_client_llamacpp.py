# llm_client_llamacpp.py
from llama_cpp import Llama
from typing import List, Dict

class LLMClient:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8):
        # Adjust n_threads to your CPU cores
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 300, temperature: float = 0.0) -> str:
        """
        messages = [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]
