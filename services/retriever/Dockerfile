FROM base

WORKDIR /app
COPY shared/ /app/shared/
COPY services/retriever/ .
RUN mkdir -p /app/data
EXPOSE ${RETRIEVER_PORT:-8002}
CMD ["python", "main.py"]