# SPDX-License-Identifier: MPL-2.0

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

FROM ${PYTHON_IMAGE} AS wheel

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE NOTICE MANIFEST.in Makefile start tsq ./
COPY src ./src
COPY corpus ./corpus
COPY benchmarks ./benchmarks
COPY scripts ./scripts

RUN python -m pip wheel --wheel-dir /wheels .


FROM wheel AS source-checks

COPY experiments ./experiments
COPY tests ./tests

RUN python scripts/sync_bundled_corpus.py \
    && PYTHONPATH=src python -m tsq audit corpus --strict \
    && PYTHONPATH=src python -W error::ResourceWarning \
        -m unittest discover -s tests -v \
    && PYTHONPATH=src python scripts/run_behavioral_audit.py \
        --root t_machine_learning --profile strong \
        --trials 5 --steps 24 --summary-only \
    && PYTHONPATH=src python scripts/run_behavioral_audit.py \
        --root t_machine_learning --profile always-wrong \
        --trials 5 --steps 24 --summary-only \
    && touch /tmp/source-checks-passed


FROM ${PYTHON_IMAGE} AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    TSQ_DB=/data/tsq.db

LABEL org.opencontainers.image.title="The Second Question" \
      org.opencontainers.image.description="Explainable knowledge-graph adaptive learning CLI" \
      org.opencontainers.image.licenses="MPL-2.0"

COPY --from=wheel /wheels /wheels
COPY LICENSE NOTICE /licenses/

RUN python -m pip install --no-index /wheels/*.whl \
    && rm -rf /wheels \
    && groupadd --gid 10001 tsq \
    && useradd --uid 10001 --gid 10001 --create-home tsq \
    && install -d -o 10001 -g 10001 /data

WORKDIR /data
USER 10001:10001
VOLUME ["/data"]

ENTRYPOINT ["tsq"]
CMD ["--help"]


FROM runtime AS runtime-smoke

RUN set -eux; \
    test -n "$(tsq --version)"; \
    tsq audit --strict --json; \
    tsq --db /tmp/tsq-smoke.db init --json; \
    tsq --db /tmp/tsq-smoke.db topics --json; \
    printf 'q\n' | tsq --db /tmp/tsq-smoke.db start \
        --topic "LLM Agents" --limit 1 --seed 7 --no-confidence; \
    tsq --db /tmp/tsq-smoke.db verify --json; \
    rm -f /tmp/tsq-smoke.db /tmp/tsq-smoke.db-shm /tmp/tsq-smoke.db-wal


FROM runtime-smoke AS test

COPY --from=source-checks /tmp/source-checks-passed /tmp/source-checks-passed

RUN test -f /tmp/source-checks-passed


FROM runtime-smoke AS release

ARG RELEASE_VERSION

RUN test -n "$RELEASE_VERSION" \
    && test "$(python -c 'import tsq; print(tsq.__version__)')" = "$RELEASE_VERSION"


FROM runtime AS final
