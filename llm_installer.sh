#!/usr/bin/env bash
# get_qwen_7b_gguf.sh — download Qwen2.5-7B-Instruct-Q4_K_M.gguf via Python API (uv/venv-safe)
set -euo pipefail

REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
FILENAME="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
TARGET_DIR="./models"
HF_TOKEN="${HF_TOKEN:-}"          # optional: pass env or --token
USE_HF_TRANSFER="${USE_HF_TRANSFER:-1}"

print_usage() {
  cat <<EOF
Usage: $0 [--dir <path>] [--filename <name.gguf>] [--token <hf_xxx>] [--no-transfer]
Examples:
  uv venv && source .venv/bin/activate && ./get_qwen_7b_gguf.sh
  uv run bash get_qwen_7b_gguf.sh --dir /opt/models
  HF_TOKEN=hf_xxx ./get_qwen_7b_gguf.sh
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) TARGET_DIR="$2"; shift 2;;
    --filename) FILENAME="$2"; shift 2;;
    --token) HF_TOKEN="$2"; shift 2;;
    --no-transfer) USE_HF_TRANSFER=0; shift 1;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

# Pick current Python (inside your uv venv if activated)
if command -v python >/dev/null 2>&1; then PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)"
else echo "ERROR: Python not found on PATH." >&2; exit 1; fi
echo "[*] Using Python: $PY"

# Ensure pip and deps
if ! "$PY" -m pip --version >/dev/null 2>&1; then "$PY" -m ensurepip --upgrade || true; fi
echo "[*] Installing huggingface_hub + hf-transfer into current env..."
"$PY" -m pip install -U "huggingface_hub>=0.23" hf-transfer >/dev/null

# Enable accelerated transfer (optional but recommended)
if [[ "${USE_HF_TRANSFER}" == "1" ]]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "[*] hf-transfer acceleration ENABLED."
else
  echo "[*] hf-transfer acceleration DISABLED."
fi

mkdir -p "${TARGET_DIR}"
echo "[*] Downloading '${FILENAME}' from '${REPO}' to '${TARGET_DIR}' ..."

# Use Python API (snapshot_download) so we don't rely on CLI binaries
"$PY" - <<PY
import os, sys
from huggingface_hub import snapshot_download
repo      = os.environ.get("REPO", "${REPO}")
filename  = os.environ.get("FILENAME", "${FILENAME}")
target    = os.environ.get("TARGET_DIR", "${TARGET_DIR}")
token     = os.environ.get("HF_TOKEN") or None
path = snapshot_download(
    repo_id=repo,
    allow_patterns=[filename],
    local_dir=target,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)
print(path)
PY

MODEL_PATH="${TARGET_DIR%/}/${FILENAME}"
if [[ -f "${MODEL_PATH}" ]]; then
  echo "[✓] Download complete: ${MODEL_PATH}"
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "${MODEL_PATH}";
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "${MODEL_PATH}"; fi
  ls -lh "${MODEL_PATH}"
else
  echo "[!] File not found at ${MODEL_PATH}. Check output above." >&2
  exit 1
fi

echo
echo "Load in Python:"
echo "  from llama_cpp import Llama"
echo "  llm = Llama(model_path='${MODEL_PATH}', n_ctx=4096)"
