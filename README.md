# RAG Pipeline

A production-grade Retrieval-Augmented Generation system you can point at any website or document collection and immediately start asking questions against. You give it URLs — it crawls them, indexes the content, and answers questions using a fully local LLM. No OpenAI key, no third-party vector database, no cloud dependency.

The system is built around one engineering constraint: every query must return in under 100ms at P99 for cached traffic, and under 800ms for the most complex multi-hop retrievals. Everything — the routing logic, caching thresholds, index selection, re-ranking — exists in service of that constraint.

---

## What it does

You point the scraper at a website. It crawls it, strips the boilerplate, and saves clean markdown to disk. The ingestion pipeline chunks that content, embeds it using a local sentence-transformer model, and writes the vectors to a FAISS index. From that point on, every query goes through the gateway:

1. The query gets embedded and checked against a Redis vector index. If a semantically similar query was asked before, the cached answer comes back in under 5ms — no retrieval, no LLM call.
2. If it's a cache miss, a rule-based classifier scores the query's complexity. Short, direct questions are simple. Questions with multiple entities, conjunctions, or sub-questions are complex.
3. Simple queries do a single FAISS search, retrieve the top 4 chunks, and send them to the LLM with a tight context window.
4. Complex queries do two rounds of retrieval — the second round uses named entities extracted from the first round's results to enrich the query. The retrieved chunks are then re-ranked with Maximal Marginal Relevance to cut redundancy before being sent to the LLM.
5. The LLM is Mistral-7B running locally via llama.cpp. It only answers from the provided context. If the answer isn't there, it says so.
6. The response gets cached in Redis with a TTL, and latency metrics are emitted to Prometheus.

The result is a system where repeated or similar questions are essentially free, simple questions are fast, and complex questions are thorough.

---

## Stack

Everything runs in Docker. Nothing calls an external API.

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector index | FAISS — Flat / IVF / HNSW selected automatically by corpus size |
| Semantic cache | Redis Stack with RediSearch vector index |
| LLM | Mistral-7B-Instruct Q4 GGUF via llama.cpp |
| Web scraping | trafilatura (boilerplate-aware content extraction) |
| Gateway | FastAPI with async routing |
| Observability | Prometheus + Grafana |
| Infrastructure | Terraform + AWS ECS Fargate |

---

## Quick start

**Prerequisites:** Docker + Docker Compose v2, ~6 GB disk, 8 GB RAM minimum (16 GB recommended for the LLM).

**1. Download the model**

```bash
chmod +x scripts/download_model.sh
./scripts/download_model.sh
```

This downloads `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4.1 GB) into `./models/`.

**2. Set your seed URLs**

Open `.env` and set the sites you want to index:

```bash
SCRAPER_SEED_URLS=https://docs.python.org/3/library/,https://realpython.com
SCRAPER_MAX_DEPTH=2
SCRAPER_MAX_PAGES_PER_SEED=50
```

**3. Build and start**

```bash
docker compose build --no-cache
docker compose up -d
```

**4. Scrape and index**

```bash
docker compose --profile scrape up scraper
```

The scraper crawls your seed URLs and automatically triggers ingestion when done. You can watch progress at `http://localhost:8003/scrape`.

**5. Ask questions**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does async/await work in Python?"}'
```

Response includes the answer, which route was taken (`cache_hit`, `simple`, or `complex`), latency, and how many chunks were used.

**6. Dashboards**

| | URL |
|---|---|
| Grafana | http://localhost:3000 — username and password in `.env` |
| Prometheus | http://localhost:9090 |
| Gateway API docs | http://localhost:8000/docs |
| Scraper API | http://localhost:8003/docs |

---

## How the scraper works

The scraper does a breadth-first crawl starting from your seed URLs. At each page it uses trafilatura to extract the main content — article body, documentation text — while discarding navigation, footers, cookie banners, and ads. Pages below `SCRAPER_MIN_CONTENT_LENGTH` characters are skipped. PDFs are saved directly for the ingestion pipeline to handle.

It respects `robots.txt` by default, enforces a per-domain request delay to avoid hammering servers, and stays on the seed domain unless you explicitly allow others. All of this is configurable via `.env` — no code changes needed.

```bash
# Trigger via API
curl -X POST http://localhost:8003/scrape \
  -H "Content-Type: application/json" \
  -d '{"seed_urls": ["https://docs.example.com"], "max_depth": 3, "auto_ingest": true}'

# Or via CLI
python services/scraper/run.py --seed https://docs.example.com --depth 3
```

---

## How routing works

```
Query arrives at gateway
  │
  ├── Embed query → check Redis KNN
  │     similarity ≥ 0.92?  →  return cached answer (~5ms)
  │
  ├── Score complexity (token count, entity count, conjunctions, sub-queries)
  │     score < 0.6  →  single-hop: FAISS top-4 → LLM  (~300ms)
  │     score ≥ 0.6  →  multi-hop:
  │                       round 1: FAISS top-8
  │                       extract named entities with spaCy
  │                       round 2: re-embed enriched query, FAISS top-4
  │                       MMR re-rank → LLM  (~800ms)
  │
  └── Store response in Redis (TTL = 1 hour) → emit Prometheus metrics
```

All thresholds (`0.92`, `0.6`, `4`, `8`) are env vars — change them in `.env` without touching code.

---

## FAISS index selection

The retriever picks the index type based on how many chunks you've ingested:

| Corpus size | Index | Behaviour |
|---|---|---|
| Under 10k chunks | `IndexFlatIP` | Exact search, always accurate |
| 10k – 1M chunks | `IndexIVFFlat` | Approximate, ~10x faster than flat |
| Over 1M chunks | `IndexHNSWFlat` | Graph-based ANN, scales to very large corpora |

The thresholds are `FAISS_FLAT_MAX_CHUNKS` and `FAISS_IVF_MAX_CHUNKS` in `.env`.

---

## Configuration

Everything is in `.env`. No values are hard-coded in the application. Key variables:

| Variable | Default | What it controls |
|---|---|---|
| `SCRAPER_SEED_URLS` | — | Sites to crawl |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | How similar a query must be to get a cache hit |
| `CACHE_TTL_SECONDS` | `3600` | How long cached answers are kept |
| `CLASSIFIER_COMPLEXITY_THRESHOLD` | `0.6` | Score above this triggers multi-hop |
| `MMR_LAMBDA` | `0.7` | Relevance vs diversity in re-ranking (higher = more relevant) |
| `LLM_MAX_TOKENS` | `512` | Max tokens the LLM generates per answer |
| `BENCHMARK_P99_CACHED_MAX_MS` | `100` | CI fails if cache-hit P99 exceeds this |
| `GRAFANA_ADMIN_PASSWORD` | `rag_grafana_pass` | Grafana login |

---

## Tests

```bash
pip install -r requirements-dev.txt

# Unit tests (no services needed)
pytest tests/ --ignore=tests/test_integration.py

# Integration tests (stack must be running)
INTEGRATION_TESTS=1 pytest tests/test_integration.py -v

# Latency benchmark — fires 500 queries, asserts P99 targets
python scripts/benchmark_latency.py

# Retrieval recall evaluation
python scripts/eval_recall.py --k 5

# Load test
locust -f tests/locustfile.py --host http://localhost:8000
```

The CI pipeline in `.github/workflows/ci.yml` runs all of this on every push and blocks merges if P99 latency regresses by more than 20%.

---

## AWS deployment

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# fill in your AWS account, region, ECR registry

terraform init && terraform apply
```

Creates: ECS Fargate cluster, ALB, ECR repos for all services, ElastiCache Redis (same VPC, sub-1ms access), CloudWatch log groups, and a P99 latency alarm that fires if response time exceeds `ALERT_P99_THRESHOLD_MS`.

---

## Project layout

```
rag-pipeline/
├── Dockerfile                 ← single multi-stage file, one pip install
├── docker-compose.yml
├── .env                       ← all config lives here
├── services/
│   ├── gateway/               ← routing, complexity classifier, MMR
│   ├── embedder/              ← sentence-transformers HTTP server
│   ├── retriever/             ← FAISS engine, adaptive index selection
│   ├── cache/                 ← Redis semantic cache
│   ├── llm/                   ← llama.cpp wrapper
│   ├── scraper/               ← web crawler + auto-ingest
│   └── ingestion/             ← chunking, embedding, indexing
├── shared/
│   └── config.py              ← all settings via Pydantic-Settings
├── infra/
│   ├── terraform/             ← AWS ECS infrastructure
│   └── grafana/               ← dashboard JSON + provisioning
├── monitoring/                ← Prometheus config, alerting rules
├── tests/                     ← unit, integration, benchmark, locust
├── scripts/                   ← benchmark, eval, model download
├── data/                      ← FAISS index + SQLite metadata (gitignored)
└── models/                    ← GGUF model files (gitignored)
```