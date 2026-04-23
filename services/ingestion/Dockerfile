FROM base

WORKDIR /app
COPY shared/ /app/shared/
COPY services/ingestion/ .
RUN mkdir -p /app/data/raw
CMD ["python", "ingest.py"]