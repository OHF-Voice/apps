ARG BUILD_FROM
FROM ${BUILD_FROM}
ARG BUILD_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src/voice

COPY requirements.txt ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && python3 -m venv /usr/src/.venv \
    && /usr/src/.venv/bin/pip3 install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/src/.venv/bin:${PATH}"

# The Kaldi and OpenFST binaries and the acoustic model are staged into
# local/ by script/stage_build rather than fetched here: they are ~255 MB
# that no package index serves. The verifier's code is not among them -- it
# is vendored in voice/kaldi/ -- so there is nothing to install, and an
# image built without local/ is a working app that verifies nothing, which
# is what a machine with no model should look like.
COPY local/ ./local/
RUN test -d local/wake-kaldi-models \
    || echo "No acoustic model staged: this image will verify nothing"

COPY wake.yaml ./
COPY voice/ ./voice/
COPY config.yaml ./

# The resident decoder daemon, built from the vendored source against the
# staged Kaldi tools. Its build-time needs are not the app's, so they are
# installed and taken away again here; libopenblas0 is asked for by name so
# that --auto-remove keeps it, because the daemon links it at runtime.
#
# Skipping this is not a build failure: without the daemon every decode
# spawns its own Kaldi process, which works. On one window of "okay nabu",
# 42 ms with the daemon against roughly 200 ms without.
RUN \
    if [ -d local/wake-kaldi-tools ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            libgfortran5 \
            libopenblas0 \
        && apt-get install -y --no-install-recommends \
            g++ \
            libopenblas-dev \
        && bash voice/kaldi/decoderd/build.sh "$PWD/local/wake-kaldi-tools" \
        && rm -rf voice/kaldi/decoderd/.fst_shadow_include \
        && apt-get purge -y --auto-remove g++ libopenblas-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* \
        && ldd voice/kaldi/decoderd/verify_decoderd > /dev/null; \
    else \
        echo "No Kaldi tools staged: decodes will each spawn a process"; \
    fi

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=2m \
    CMD curl -sf http://127.0.0.1:10800/health > /dev/null || exit 1
