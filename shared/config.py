"""
shared/config.py — Centralised Pydantic-Settings configuration.
Every service imports the subset it needs from here.
All values are read from environment variables (set via .env or Docker).
Nothing is hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(384, alias="EMBEDDING_DIM")
    batch_size: int = Field(512, alias="EMBEDDING_BATCH_SIZE")
    # bind_host: what this service binds to inside its own container (always 0.0.0.0).
    # EMBEDDER_HOST is the hostname OTHER services use to reach this one — never use it here.
    bind_host: str = Field("0.0.0.0", alias="EMBEDDER_BIND_HOST")
    port: int = Field(8001, alias="EMBEDDER_PORT")
    single_latency_target_ms: float = Field(8.0, alias="EMBEDDING_SINGLE_LATENCY_TARGET_MS")
    batch_latency_target_ms: float = Field(180.0, alias="EMBEDDING_BATCH_LATENCY_TARGET_MS")


class RetrieverSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    faiss_index_path: str = Field("/app/data/index.faiss", alias="FAISS_INDEX_PATH")
    sqlite_path: str = Field("/app/data/index_meta.sqlite", alias="SQLITE_PATH")
    flat_max_chunks: int = Field(10_000, alias="FAISS_FLAT_MAX_CHUNKS")
    ivf_max_chunks: int = Field(1_000_000, alias="FAISS_IVF_MAX_CHUNKS")
    ivf_nprobe: int = Field(32, alias="FAISS_IVF_NPROBE")
    hnsw_m: int = Field(32, alias="FAISS_HNSW_M")
    hnsw_ef_search: int = Field(64, alias="FAISS_HNSW_EF_SEARCH")
    default_k: int = Field(4, alias="RETRIEVER_DEFAULT_K")
    simple_k: int = Field(4, alias="RETRIEVER_SIMPLE_K")
    complex_k: int = Field(12, alias="RETRIEVER_COMPLEX_K")
    multihop_k1: int = Field(8, alias="RETRIEVER_MULTIHOP_K1")
    multihop_k2: int = Field(4, alias="RETRIEVER_MULTIHOP_K2")
    # bind_host: what this service binds to inside its own container.
    # RETRIEVER_HOST is for OTHER services to reach this one.
    bind_host: str = Field("0.0.0.0", alias="RETRIEVER_BIND_HOST")
    port: int = Field(8002, alias="RETRIEVER_PORT")


class CacheSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_host: str = Field("redis-stack", alias="CACHE_HOST")
    redis_port: int = Field(6379, alias="CACHE_PORT")
    similarity_threshold: float = Field(0.92, alias="CACHE_SIMILARITY_THRESHOLD")
    ttl_seconds: int = Field(3600, alias="CACHE_TTL_SECONDS")
    index_name: str = Field("idx:cache", alias="CACHE_INDEX_NAME")
    key_prefix: str = Field("rag:cache:", alias="CACHE_KEY_PREFIX")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = Field("llm", alias="LLM_HOST")
    port: int = Field(8080, alias="LLM_PORT")
    context_size: int = Field(4096, alias="LLM_CONTEXT_SIZE")
    n_gpu_layers: int = Field(0, alias="LLM_N_GPU_LAYERS")
    max_tokens: int = Field(512, alias="LLM_MAX_TOKENS")
    temperature: float = Field(0.1, alias="LLM_TEMPERATURE")
    stream: bool = Field(False, alias="LLM_STREAM")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ClassifierSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    complexity_threshold: float = Field(0.6, alias="CLASSIFIER_COMPLEXITY_THRESHOLD")
    token_weight: float = Field(0.3, alias="CLASSIFIER_TOKEN_WEIGHT")
    entity_weight: float = Field(0.3, alias="CLASSIFIER_ENTITY_WEIGHT")
    conjunction_weight: float = Field(0.2, alias="CLASSIFIER_CONJUNCTION_WEIGHT")
    subquery_weight: float = Field(0.2, alias="CLASSIFIER_SUBQUERY_WEIGHT")
    max_tokens_score: int = Field(50, alias="CLASSIFIER_MAX_TOKENS_SCORE")


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bind_host: str = Field("0.0.0.0", alias="GATEWAY_BIND_HOST")
    port: int = Field(8000, alias="GATEWAY_PORT")

    embedder_host: str = Field("embedder", alias="EMBEDDER_HOST")
    embedder_port: int = Field(8001, alias="EMBEDDER_PORT")
    retriever_host: str = Field("retriever", alias="RETRIEVER_HOST")
    retriever_port: int = Field(8002, alias="RETRIEVER_PORT")
    cache_host: str = Field("redis-stack", alias="CACHE_HOST")
    cache_port: int = Field(6379, alias="CACHE_PORT")
    llm_host: str = Field("llm", alias="LLM_HOST")
    llm_port: int = Field(8080, alias="LLM_PORT")

    cache_hit_target_ms: float = Field(5.0, alias="GATEWAY_CACHE_HIT_TARGET_MS")
    simple_target_ms: float = Field(300.0, alias="GATEWAY_SIMPLE_TARGET_MS")
    complex_target_ms: float = Field(800.0, alias="GATEWAY_COMPLEX_TARGET_MS")

    mmr_lambda: float = Field(0.7, alias="MMR_LAMBDA")

    @property
    def embedder_url(self) -> str:
        return f"http://{self.embedder_host}:{self.embedder_port}"

    @property
    def retriever_url(self) -> str:
        return f"http://{self.retriever_host}:{self.retriever_port}"

    @property
    def llm_url(self) -> str:
        return f"http://{self.llm_host}:{self.llm_port}"


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    data_dir: str = Field("/app/data/raw", alias="INGEST_DATA_DIR")
    faiss_index_path: str = Field("/app/data/index.faiss", alias="FAISS_INDEX_PATH")
    sqlite_path: str = Field("/app/data/index_meta.sqlite", alias="SQLITE_PATH")
    embedder_host: str = Field("embedder", alias="EMBEDDER_HOST")
    embedder_port: int = Field(8001, alias="EMBEDDER_PORT")
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".md", ".txt"],
        alias="INGEST_SUPPORTED_EXTENSIONS",
    )

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v):
        if isinstance(v, str):
            return [e.strip() for e in v.split(",")]
        return v

    @property
    def embedder_url(self) -> str:
        return f"http://{self.embedder_host}:{self.embedder_port}"


@lru_cache
def get_embedder_settings() -> EmbedderSettings:
    return EmbedderSettings()


@lru_cache
def get_retriever_settings() -> RetrieverSettings:
    return RetrieverSettings()


@lru_cache
def get_cache_settings() -> CacheSettings:
    return CacheSettings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


@lru_cache
def get_classifier_settings() -> ClassifierSettings:
    return ClassifierSettings()


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


class ScraperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    # Seed URLs — comma-separated list of URLs to start crawling from
    seed_urls: List[str] = Field(
        default_factory=list,
        alias="SCRAPER_SEED_URLS",
    )
    # Maximum pages to crawl per seed URL (0 = unlimited)
    max_pages_per_seed: int = Field(50, alias="SCRAPER_MAX_PAGES_PER_SEED")
    # Maximum crawl depth (1 = seed page only, 2 = seed + linked pages, etc.)
    max_depth: int = Field(2, alias="SCRAPER_MAX_DEPTH")
    # Only follow links on the same domain as the seed URL
    stay_on_domain: bool = Field(True, alias="SCRAPER_STAY_ON_DOMAIN")
    # Additional allowed domains (comma-separated), ignored if stay_on_domain=False
    allowed_domains: List[str] = Field(default_factory=list, alias="SCRAPER_ALLOWED_DOMAINS")
    # URL patterns to skip (comma-separated regex strings)
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [r"\.(jpg|jpeg|png|gif|svg|ico|css|js|woff|woff2|ttf|eot|pdf|zip|tar|gz)$"],
        alias="SCRAPER_EXCLUDE_PATTERNS",
    )
    # Include these URL patterns even if they'd otherwise be excluded (e.g. PDFs you DO want)
    include_patterns: List[str] = Field(default_factory=list, alias="SCRAPER_INCLUDE_PATTERNS")
    # Request timeout per page (seconds)
    request_timeout_s: float = Field(15.0, alias="SCRAPER_REQUEST_TIMEOUT_S")
    # Delay between requests to the same domain (seconds) — be polite
    request_delay_s: float = Field(0.5, alias="SCRAPER_REQUEST_DELAY_S")
    # Concurrent requests
    concurrency: int = Field(5, alias="SCRAPER_CONCURRENCY")
    # User-agent string
    user_agent: str = Field(
        "RAGPipelineBot/1.0 (+https://github.com/your-repo/rag-pipeline)",
        alias="SCRAPER_USER_AGENT",
    )
    # Output directory for scraped content
    output_dir: str = Field("/app/data/raw", alias="SCRAPER_OUTPUT_DIR")
    # Save format: "markdown" or "text"
    output_format: str = Field("markdown", alias="SCRAPER_OUTPUT_FORMAT")
    # Respect robots.txt
    respect_robots_txt: bool = Field(True, alias="SCRAPER_RESPECT_ROBOTS_TXT")
    # Minimum content length to save (chars) — filters nav/footer-only pages
    min_content_length: int = Field(200, alias="SCRAPER_MIN_CONTENT_LENGTH")
    # Whether to auto-trigger ingestion after scraping completes
    auto_ingest: bool = Field(True, alias="SCRAPER_AUTO_INGEST")
    # Embedder URL for auto-ingest
    embedder_host: str = Field("embedder", alias="EMBEDDER_HOST")
    embedder_port: int = Field(8001, alias="EMBEDDER_PORT")

    @field_validator("seed_urls", "allowed_domains", "exclude_patterns", "include_patterns", mode="before")
    @classmethod
    def parse_csv(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v or []

    @property
    def embedder_url(self) -> str:
        return f"http://{self.embedder_host}:{self.embedder_port}"


@lru_cache
def get_scraper_settings() -> ScraperSettings:
    return ScraperSettings()


@lru_cache
def get_ingestion_settings() -> IngestionSettings:
    return IngestionSettings()