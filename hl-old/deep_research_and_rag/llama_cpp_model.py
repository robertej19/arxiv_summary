# local_corpus/llama_cpp_model.py
"""
Smolagents-compatible wrapper for llama-cpp-python.

Features:
- Accepts dict/Pydantic-like messages; flattens multimodal content to text.
- Returns a SimpleMessage with .content and .token_usage (.input_tokens/.output_tokens).
- Auto-truncates chat history to fit within context window.
- Compatible with multiple llama_cpp APIs (create_chat_completion / chat_completion / OpenAI-like).

Usage:
    model = LlamaCppModel(
        model_path="path/to/model.gguf",
        n_ctx=8192,          # raise to 32768 if your model/VRAM permits
        max_tokens=512,      # safe default to avoid overflows
        temperature=0.2,
    )
    msg = model.generate(messages, stop_sequences=["<end>"])
    print(msg.content)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError("Install llama-cpp-python: pip install llama-cpp-python") from e

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MessageDict = Dict[str, Any]
Messages = List[Union[MessageDict, Any]]  # allow Pydantic-like objects

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: Optional[int] = None

@dataclass
class SimpleMessage:
    role: str
    content: str
    name: Optional[str] = None
    token_usage: Optional[TokenUsage] = None

# Leave some buffer for control/template tokens
TRIM_SAFETY = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_content(content: Any) -> str:
    """Normalize message content to a single text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(str(p["text"]))
                # ignore non-text parts (image/audio) for llama.cpp
            else:
                parts.append(str(p))
        return "\n".join(s for s in parts if s)
    return str(content)


def _convert_messages(messages: Messages) -> List[Dict[str, str]]:
    """
    Convert mixed messages into [{role, content, (name)}] for llama.cpp.
    Coerces unknown roles and function/tool messages to supported roles.
    """
    chat: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = _normalize_content(m.get("content", ""))
            name = m.get("name")
            ttype = m.get("type")
        else:
            role = getattr(m, "role", "user")
            content_attr = (
                getattr(m, "content", None)
                or getattr(m, "text", None)
                or getattr(m, "message", None)
            )
            content = _normalize_content(content_attr)
            name = getattr(m, "name", None)
            ttype = getattr(m, "type", None)

        if role not in ("system", "user", "assistant", "tool"):
            # Some frameworks use function/tool_call/function_call
            if (ttype and ttype in ("tool", "function")) or role in (
                "function",
                "tool_call",
                "function_call",
            ):
                role = "tool"
            else:
                role = "user"

        payload = {"role": role, "content": content}
        if name and isinstance(name, str):
            payload["name"] = name
        chat.append(payload)
    return chat


def _to_token_usage(u: Optional[Dict[str, Any]]) -> Optional[TokenUsage]:
    """Map various usage dict shapes to TokenUsage (attribute access)."""
    if not u:
        return None
    prompt = int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
    completion = int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    total = u.get("total_tokens")
    if total is None:
        total = prompt + completion if (prompt or completion) else None
    return TokenUsage(input_tokens=prompt, output_tokens=completion, total_tokens=total)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class LlamaCppModel:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        seed: Optional[int] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        max_tokens: int = 512,      # safer default to avoid context overflow
        verbose: bool = False,
        **llama_kwargs: Any,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.default_max_tokens = max_tokens
        self.n_ctx = n_ctx  # keep our configured n_ctx

        init_kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )
        init_kwargs.update(llama_kwargs)

        self.client = Llama(**init_kwargs)

        # Probe API variants once
        self._has_create_chat_completion = hasattr(self.client, "create_chat_completion")
        self._has_chat_completion = hasattr(self.client, "chat_completion")
        self._has_openai_style = hasattr(getattr(self.client, "chat", None), "completions")

    # ---------- Token counting & truncation ----------

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Approximate token count using llama.cpp tokenizer on a simple serialized chat.
        TRIM_SAFETY covers template/control-token differences.
        """
        serialized = []
        for m in messages:
            serialized.append(f"{m.get('role','user')}: {m.get('content','')}\n")
        text = "".join(serialized)
        try:
            return len(self.client.tokenize(text.encode("utf-8")))
        except Exception:
            # Fallback heuristic if tokenizer unavailable
            return max(1, len(text) // 4)

    def _truncate_chat_to_fit(
        self,
        messages: List[Dict[str, str]],
        n_ctx: int,
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """
        Drop oldest non-system messages until prompt fits:
        prompt_tokens + max_tokens + TRIM_SAFETY <= n_ctx
        """
        msgs = list(messages)
        # Nothing to do if empty
        if not msgs:
            return msgs

        while True:
            used = self._count_tokens(msgs)
            if used + max_tokens + TRIM_SAFETY <= n_ctx:
                return msgs

            # Find oldest message that is not a system message to drop
            drop_idx = None
            for i, m in enumerate(msgs):
                if m.get("role") != "system":
                    drop_idx = i
                    break
            if drop_idx is None:
                # We cannot drop anything else; return as-is (will likely overflow but avoids infinite loop)
                return msgs
            del msgs[drop_idx]

    # ---------- Core call ----------

    def _call_chat_raw(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        stop = stop or []
        max_tokens = max_tokens or self.default_max_tokens
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        repeat_penalty = self.repeat_penalty if repeat_penalty is None else repeat_penalty

        # Trim to fit the configured context window
        n_ctx = self.n_ctx
        messages = self._truncate_chat_to_fit(messages, n_ctx=n_ctx, max_tokens=max_tokens)

        common = dict(
            messages=messages,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
        common.update(kwargs)

        usage: Optional[Dict[str, int]] = None

        if self._has_create_chat_completion:
            out = self.client.create_chat_completion(**common)
            text = out["choices"][0]["message"]["content"]
            u = out.get("usage") or {}
            usage = {k: int(u.get(k, 0)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if u else None
            return text, usage

        if self._has_chat_completion:
            out = self.client.chat_completion(**common)
            if isinstance(out, dict):
                msg = out.get("message") or {}
                text = msg.get("content", "") or ""
                u = out.get("usage") or {}
                usage = {k: int(u.get(k, 0)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if u else None
                return text, usage
            return str(out), None

        if self._has_openai_style:
            out = self.client.chat.completions.create(**common)
            choice0 = out.choices[0]
            text = getattr(choice0.message, "content", "") or ""
            if hasattr(out, "usage") and out.usage:
                u = out.usage
                usage = {
                    "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
                }
            return text, usage

        raise RuntimeError("llama_cpp.Llama does not expose a known chat completion method.")

    # ---------- Smolagents interface ----------

    def generate(
        self,
        messages: Messages,
        stop_sequences: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> SimpleMessage:
        """
        Return a message object with .content and .token_usage (attributes),
        so smolagents can access plan_message.content and plan_message.token_usage.input_tokens.
        """
        chat = _convert_messages(messages)
        stop = list(stop_sequences) if stop_sequences else None

        if not chat:
            chat = [{"role": "user", "content": " "}]
        elif chat[-1]["role"] == "assistant" and not chat[-1]["content"].strip():
            chat[-1]["content"] = " "

        text, usage_dict = self._call_chat_raw(chat, stop=stop, **kwargs)
        usage_obj = _to_token_usage(usage_dict)
        return SimpleMessage(role="assistant", content=text, token_usage=usage_obj)

    def __call__(self, messages: Messages, **kwargs: Any) -> SimpleMessage:
        return self.generate(messages, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llama_cpp_model.py /path/to/model.gguf")
        sys.exit(1)

    model_path = sys.argv[1]
    model = LlamaCppModel(model_path=model_path, n_ctx=8192, max_tokens=256, temperature=0.2)

    msgs = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
    msg = model.generate(msgs, stop_sequences=["</s>"])
    print("ASSISTANT:", msg.content, "| usage:", msg.token_usage)
