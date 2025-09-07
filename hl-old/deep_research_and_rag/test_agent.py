from deep_research_and_rag.llama_cpp_model import LlamaCppModel
m = LlamaCppModel(model_path="../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", n_ctx=4096)
msg = [{"role":"user","content":[{"type":"text","text":"Say 'ready' in one word."}]}]
print(m(msg, stop_sequences=["\n"]).content)
