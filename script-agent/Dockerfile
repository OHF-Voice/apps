# Build llama-cpp-python separately to avoid rebuilding.
# We always compile from source to maximize optimizations.
FROM ghcr.io/home-assistant/base-debian:trixie AS llama

ARG TARGETARCH

WORKDIR /usr/src

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        cmake \
        ninja-build

RUN \
    python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade \
        wheel setuptools

COPY requirements.llama.txt ./

RUN \
    llama_cmake_args='-DCMAKE_BUILD_TYPE=Release' \
    && if [[ "${TARGETARCH}" == "arm64" ]]; then \
        llama_cmake_args+=' -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod'; \
    fi \
    && CMAKE_ARGS="${llama_cmake_args}" \
    .venv/bin/pip install --no-cache-dir \
    -r ./requirements.llama.txt

# -----------------------------------------------------------------------------

FROM ghcr.io/home-assistant/base-debian:trixie

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install
WORKDIR /usr/src

COPY --from=llama /usr/src/.venv/ ./.venv/

COPY requirements.txt ./

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
        libgomp1 \
    && .venv/bin/pip install --no-cache-dir \
        -r ./requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY src/*.py ./
COPY src/benchmark.yaml ./
COPY src/templates/ ./templates/

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=10m \
    CMD curl -f http://localhost:5000/health || exit 1
