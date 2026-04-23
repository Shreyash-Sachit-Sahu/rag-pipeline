# ── Stage 1: base — pip runs ONCE here, all services inherit it ──────────────
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "httpx>=0.27.0" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.2.0" \
    "numpy>=1.26.0" \
    "prometheus-client>=0.20.0" \
    "sentence-transformers>=2.7.0" \
    "faiss-cpu>=1.8.0" \
    "spacy>=3.7.0" \
    "redis[hiredis]>=5.0.0" \
    "trafilatura>=1.12.0" \
    "langchain-community>=0.2.0" \
    "pypdf>=4.0.0" \
    "unstructured>=0.14.0" \
    && python -m spacy download en_core_web_sm

COPY shared/ /app/shared/


# ── Stage 2: embedder ─────────────────────────────────────────────────────────
FROM base AS embedder

WORKDIR /app
COPY services/embedder/ .
RUN mkdir -p /app/data
EXPOSE ${EMBEDDER_PORT:-8001}
CMD ["python", "main.py"]




# ── Stage 3: retriever ────────────────────────────────────────────────────────
FROM base AS retriever

WORKDIR /app
COPY services/retriever/ .
RUN mkdir -p /app/data
EXPOSE ${RETRIEVER_PORT:-8002}
CMD ["python", "main.py"]


# ── Stage 4: gateway ──────────────────────────────────────────────────────────
FROM base AS gateway

WORKDIR /app
COPY services/gateway/ .
COPY services/cache/cache_client.py /app/cache_client.py
COPY services/llm/client.py /app/llm_client.py
EXPOSE ${GATEWAY_PORT:-8000}
CMD ["python", "main.py"]


# ── Stage 5: scraper ──────────────────────────────────────────────────────────
FROM base AS scraper

WORKDIR /app
COPY services/scraper/ .
COPY services/ingestion/ingest.py /app/ingest.py
RUN mkdir -p /app/data/raw
EXPOSE ${SCRAPER_PORT:-8003}
CMD ["python", "main.py"]


# ── Stage 6: ingestion ────────────────────────────────────────────────────────
FROM base AS ingestion

WORKDIR /app
COPY services/ingestion/ .
RUN mkdir -p /app/data/raw
CMD ["python", "ingest.py"]


# ── Stage 7: llm (separate — needs C++ build for llama.cpp) ──────────────────
FROM python:3.11-slim AS llm-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# BUILD_SHARED_LIBS=OFF → fully static binary, no shared lib deps
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" && \
    cmake --build build --config Release -j$(nproc) --target llama-server

FROM python:3.11-slim AS llm

RUN apt-get update && apt-get install -y --no-install-recommends \  
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Only copy the single static binary — no shared libs needed
COPY --from=llm-build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

RUN pip install --no-cache-dir --timeout=120 --retries=5 \
    "httpx>=0.27.0" "pydantic>=2.7.0" "pydantic-settings>=2.2.0"

WORKDIR /app
COPY shared/ /app/shared/
COPY services/llm/ .
RUN mkdir -p /app/models
COPY services/llm/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE ${LLM_PORT:-8080}
ENTRYPOINT ["/entrypoint.sh"]