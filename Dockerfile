# ── Build stage ──
# Pin Python to specific patch version for reproducibility
# NOTE: for maximum reproducibility, pin to a SHA digest:
#   python:3.12.8-slim@sha256:<digest>
FROM python:3.14.0-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY constraints.txt ./
COPY src ./src

RUN pip install --no-cache-dir --prefix=/install -c constraints.txt .

# ── Runtime stage ──
# .dockerignore should exclude: .git, .venv, __pycache__, *.pyc, .env, .pytest_cache, data/
FROM python:3.14.0-slim

LABEL maintainer="polymarket-trader-team"
LABEL version="1.0.0"
LABEL description="Polymarket Neural Trading Agent"

# Security: run as non-root
RUN groupadd -r trader && useradd -r -g trader -m trader

# Install tini for proper signal handling / PID 1
RUN apt-get update && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY config ./config
COPY scripts ./scripts
COPY backtest ./backtest
COPY dashboard ./dashboard

# Volume mount points for persistent data and logs
VOLUME ["/app/data", "/app/logs"]

# Create data directories owned by non-root user
RUN mkdir -p data/logs data/models data/price_history \
 && chown -R trader:trader /app

USER trader

ENV PYTHONUNBUFFERED=1

EXPOSE 8501 8080

# Healthcheck using the dedicated healthcheck script
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/healthcheck.py || exit 1

# Graceful shutdown
STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
