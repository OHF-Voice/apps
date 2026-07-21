ARG BUILD_FROM
FROM ${BUILD_FROM}
ARG BUILD_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt ./

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        netcat-traditional \
        python3 \
        python3-pip \
        git \
        libfst-dev \
        libfst-tools \
        cmake \
        g++ \
    && pip3 install --no-cache-dir -U setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Add-on application + curated sentences + intents metadata
COPY src/ ./src/
COPY sentences/ ./sentences/
COPY intents.yaml ./intents.yaml

WORKDIR /
COPY rootfs /

# The Wyoming server answers a describe with the add-on name.
HEALTHCHECK --start-period=15m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10300 \
        | grep -iq "speech-to-phrase" \
        || exit 1
