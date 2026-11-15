# CPU image: front end, dataset preparation, annotation and inference.
# Training is not included: it needs a GPU and a much larger image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ZERRE_DATA=/data \
    ZERRE_WEIGHTS=/weights \
    ZERRE_RUNS=/runs

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[prepare]"

COPY scripts ./scripts
COPY tests ./tests

RUN useradd -m -u 1000 zerre && mkdir -p /data /weights /runs && chown -R zerre /data /weights /runs
USER zerre

ENTRYPOINT ["python", "-m"]
CMD ["zerre.infer", "--help"]
