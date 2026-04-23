FROM base

WORKDIR /app
COPY shared/ /app/shared/
COPY services/scraper/ .
COPY services/ingestion/ingest.py /app/ingest.py
RUN mkdir -p /app/data/raw
EXPOSE ${SCRAPER_PORT:-8003}
CMD ["python", "main.py"]