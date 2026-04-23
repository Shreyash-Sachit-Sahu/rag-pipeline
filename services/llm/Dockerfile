FROM python:3.11-slim AS llm-build

RUN apt-get update && apt-get install -y --no-install-recommends     curl git cmake build-essential     && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp &&     cd /opt/llama.cpp &&     cmake -B build -DLLAMA_CURL=OFF &&     cmake --build build --config Release -j$(nproc) --target llama-server &&     cp build/bin/llama-server /usr/local/bin/llama-server

# Final stage — only copy the binary, skip all build deps
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl     && rm -rf /var/lib/apt/lists/*

COPY --from=llm-build /usr/local/bin/llama-server /usr/local/bin/llama-server

RUN pip install --no-cache-dir --timeout=120 --retries=5     httpx>=0.27.0 pydantic>=2.7.0 pydantic-settings>=2.2.0

WORKDIR /app
COPY shared/ /app/shared/
COPY services/llm/ .
RUN mkdir -p /app/models
COPY services/llm/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE ${LLM_PORT:-8080}
ENTRYPOINT ["/entrypoint.sh"]