ARG BUILD_FROM
FROM ${BUILD_FROM}
ARG BUILD_ARCH

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV DEFAULT_AGENT_VERSION=0.1.0

COPY wheels/*.whl /wheels/

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        netcat-traditional \
        python3 \
        python3-pip \
        git \
    && pip install --no-cache-dir -U \
        setuptools \
        wheel \
    && pip install --no-cache-dir \
        "default-agent[zeroconf]@https://github.com/OHF-Voice/default-agent/archive/refs/tags/v${DEFAULT_AGENT_VERSION}.tar.gz" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10500 \
        | grep -iq "default-agent" \
        || exit 1
