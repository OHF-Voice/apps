FROM ghcr.io/home-assistant/base-debian:bookworm

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install
WORKDIR /usr/src
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt ./

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        cmake \
        ninja-build \
    && CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release' pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && pip3 install --no-cache-dir -r ./requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY ./tools.yaml ./
COPY src/*.py ./

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=10m \
    CMD curl -f http://localhost:5000/health || exit 1
