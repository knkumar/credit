FROM python:3.11-slim-bookworm AS source

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTENSOR_FLAGS="cxx=" \
    UV_LINK_MODE=copy

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY calmmm/ ./calmmm/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY outputs/calmmm_sample_weekly_panel.csv outputs/calmmm_sample_lift_tests.csv ./outputs/

FROM source AS test

RUN uv sync --frozen --dev

CMD ["pytest", "-v"]

FROM source AS runtime

RUN uv sync --frozen --no-dev \
    && mkdir -p artifacts/demo_fit reporting

ENTRYPOINT ["python", "scripts/run_demo_fit.py"]
CMD ["--mode", "map"]
