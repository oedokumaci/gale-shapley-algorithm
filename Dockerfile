# Base images are pinned by digest; Dependabot's docker ecosystem keeps tags and digests current
# (it only updates FROM lines, which is why uv gets its own stage instead of COPY --from=<image>).
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

FROM node:24-slim@sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6 AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM python:3.13-slim@sha256:3dd7cc108ec1493442514f5c2a871af6af0ec31d768ff6e378a93340c3b3db5f AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Runtime dependencies only (no test/docs groups), installed before the source is copied
# so that code changes reuse this layer. --locked fails the build if uv.lock is stale.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --extra cli --extra gui --no-install-project
COPY README.md LICENSE ./
COPY src/ src/
# UV_COMPILE_BYTECODE skips the editable project itself, and the runtime user can't write
# __pycache__ next to it; hash-checked .pyc files stay valid whatever mtimes COPY gives the sources
# (-f rewrites any timestamp-based .pyc that slipped into the build context).
RUN uv sync --locked --no-default-groups --extra cli --extra gui \
    && .venv/bin/python -m compileall -q -f --invalidation-mode checked-hash src

FROM python:3.13-slim@sha256:3dd7cc108ec1493442514f5c2a871af6af0ec31d768ff6e378a93340c3b3db5f

ARG CREATED
ARG VERSION
ARG REVISION

LABEL org.opencontainers.image.source="https://github.com/oedokumaci/gale-shapley-algorithm" \
      org.opencontainers.image.description="A Python implementation of the Gale-Shapley Algorithm" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.revision="${REVISION}"

RUN useradd --system --uid 10001 --no-create-home app

WORKDIR /app
# The project is installed in editable mode, so _api/app.py finds the UI at /app/frontend/dist.
COPY --from=builder /app /app
COPY --from=frontend /app/frontend/dist frontend/dist
ENV PATH="/app/.venv/bin:$PATH"

# Numeric so that orchestrators (e.g. Kubernetes runAsNonRoot) can verify it is not root.
USER 10001
EXPOSE 8000
# Proxies disabled: urllib would otherwise route the probe through HTTP_PROXY if one is set.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/api/health', timeout=2)"]

# Run uvicorn directly: `uv run` would re-sync and rebuild the project on every start.
CMD ["uvicorn", "gale_shapley_algorithm._api.app:app", "--host", "0.0.0.0", "--port", "8000"]
