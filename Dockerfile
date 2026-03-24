FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HELIX_WORKSPACE_ROOT=/app/workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[hf,agent]"

EXPOSE 8080

CMD ["python", "-m", "helix_proto.server"]
