# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency manifests first for layer caching
COPY pyproject.toml poetry.lock* ./

# Install runtime dependencies only, no virtual env (we'll copy the wheel)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy source and build wheel
COPY gaming_highlight_gen/ gaming_highlight_gen/
COPY game_configs/ game_configs/
RUN poetry build --format wheel


# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install FFmpeg via apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install wheel produced by the builder stage
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy game configs so the CLI can locate them
COPY --from=builder /build/game_configs/ /app/game_configs/

# Runtime directories
RUN mkdir -p /app/input /app/output && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["highlight-gen"]
