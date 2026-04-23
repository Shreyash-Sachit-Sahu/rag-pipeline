#!/bin/sh
# services/llm/entrypoint.sh
#
# Starts the llama.cpp HTTP server using only environment variables.
# Every parameter is configurable — nothing is hard-coded.
# Required env vars (set in .env / docker-compose):
#   LLM_MODEL_PATH   — path to the GGUF model file
#   LLM_PORT         — port to listen on
#   LLM_CONTEXT_SIZE — context window in tokens
#   LLM_N_GPU_LAYERS — number of layers to offload to GPU (0 = CPU only)

set -e

MODEL="${LLM_MODEL_PATH:-/app/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"
PORT="${LLM_PORT:-8080}"
CONTEXT="${LLM_CONTEXT_SIZE:-4096}"
GPU_LAYERS="${LLM_N_GPU_LAYERS:-0}"

if [ ! -f "${MODEL}" ]; then
    echo "[llm] ERROR: Model file not found at ${MODEL}"
    echo "[llm] Download it with:"
    echo "  ./scripts/download_model.sh"
    exit 1
fi

echo "[llm] Starting llama-server"
echo "  model       = ${MODEL}"
echo "  port        = ${PORT}"
echo "  context     = ${CONTEXT}"
echo "  gpu_layers  = ${GPU_LAYERS}"

exec llama-server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --ctx-size "${CONTEXT}" \
    --n-gpu-layers "${GPU_LAYERS}" \
    --log-disable
