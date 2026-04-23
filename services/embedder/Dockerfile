FROM base

WORKDIR /app
COPY shared/ /app/shared/
COPY services/embedder/ .
RUN mkdir -p /app/data
EXPOSE ${EMBEDDER_PORT:-8001}
CMD ["python", "main.py"]