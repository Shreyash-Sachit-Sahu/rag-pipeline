FROM base

WORKDIR /app
COPY shared/ /app/shared/
COPY services/gateway/ .
COPY services/cache/cache_client.py /app/cache_client.py
COPY services/llm/client.py /app/llm_client.py
EXPOSE ${GATEWAY_PORT:-8000}
CMD ["python", "main.py"]