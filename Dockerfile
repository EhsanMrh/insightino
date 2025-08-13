FROM python:3.11-slim

ARG QDRANT_VERSION=v1.7.3
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Qdrant binary
RUN mkdir -p /opt/qdrant /qdrant/storage && \
    set -ex && \
    (curl -L -o /tmp/qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz || \
     curl -L -o /tmp/qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${QDRANT_VERSION#v}-x86_64-unknown-linux-gnu.tar.gz) && \
    tar -xzf /tmp/qdrant.tar.gz -C /opt/qdrant --strip-components=1 && \
    ln -s /opt/qdrant/qdrant /usr/local/bin/qdrant && \
    rm -f /tmp/qdrant.tar.gz

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app code and entrypoint
COPY . /app
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Defaults (internal Qdrant, only UI port exposed)
ENV QDRANT_URL=http://127.0.0.1:6333 \
    QDRANT_COLLECTION_TEXT=insightino_text \
    PORT=7860

EXPOSE 7860
ENTRYPOINT ["/app/docker/entrypoint.sh"]


