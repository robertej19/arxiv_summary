# local_corpus/agent_demo.py
from __future__ import annotations
import os
from smolagents import CodeAgent
from tools_local import LocalSearchTool, ReaderTool
from llama_cpp_model import LlamaCppModel

INSTRUCTIONS = (
    "You are a careful research assistant.\n"
    "Plan first, then decide which tools to call.\n"
    "Every factual claim must include inline citations using the 'format' string returned by tools.\n"
    "If evidence is thin or conflicting, say so and search again."
)

def run():
    tools = [LocalSearchTool(), ReaderTool()]
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

    model = LlamaCppModel(
        model_path=model_path,
        n_ctx=8192,        # bump if you have RAM
        n_threads=None,    # or set to os.cpu_count()-1
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        repeat_penalty=1.1,
    )

    agent = CodeAgent(
        tools=tools,
        model=model,                # <- no special tool-calling needed
        instructions=INSTRUCTIONS,  # <- use 'instructions', not 'system_prompt'
        max_steps=8,
        executor_type="local",
        planning_interval=3,        # optional: add periodic planning
    )

    q = "Summarize why PPO is more practical than TRPO. Provide at least 2 citations."
    print(agent.run(q))

if __name__ == "__main__":
    run()
