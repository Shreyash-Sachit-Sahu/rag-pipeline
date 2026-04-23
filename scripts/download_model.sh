#!/usr/bin/env bash
# scripts/download_model.sh
#
# Downloads the Mistral-7B GGUF model for local LLM inference.
# All paths and URLs are configurable via environment variables.
#
# Usage:
#   ./scripts/download_model.sh
#   MODEL_DIR=/custom/path ./scripts/download_model.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$(dirname "$0")/../models}"
MODEL_FILE="${MODEL_FILE:-mistral-7b-instruct-v0.2.Q4_K_M.gguf}"
# Hugging Face repo; override if you have a mirror
HF_REPO="${HF_REPO:-TheBloke/Mistral-7B-Instruct-v0.2-GGUF}"
HF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${MODEL_FILE}"

mkdir -p "${MODEL_DIR}"
DEST="${MODEL_DIR}/${MODEL_FILE}"

if [ -f "${DEST}" ]; then
    echo "[download] Model already present at ${DEST} — skipping."
    exit 0
fi

echo "[download] Downloading ${MODEL_FILE} (~4.1 GB)…"
echo "  Source : ${HF_URL}"
echo "  Dest   : ${DEST}"

if command -v wget &>/dev/null; then
    wget --progress=bar:force -O "${DEST}" "${HF_URL}"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${DEST}" "${HF_URL}"
else
    echo "[download] ERROR: wget or curl is required." && exit 1
fi

echo "[download] Done. Model saved to ${DEST}"
