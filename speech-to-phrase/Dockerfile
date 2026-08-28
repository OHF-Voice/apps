ARG BUILD_FROM
FROM ${BUILD_FROM}
ARG BUILD_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt ./
# The recognition library is vendored (lib/) and built from these sources, so
# the image needs no network access to a git host and no external checkout.
COPY lib/ ./lib/

# cmake/g++/libfst-dev/python3-dev exist only to compile the vendored library's
# native _fst module, and are purged in the same layer so they never reach the
# shipped image (~200 MB of toolchain). python3-dev is what CMake's
# Development.SABIModule component looks for; without it the configure step
# fails with "Could NOT find Python (missing: Interpreter
# Development.SABIModule)" rather than anything about headers.
#
# The OpenFST *runtime* library has to survive that purge: _fst links
# libfst.so.<N> dynamically, and only the -dev package was asked for by name, so
# apt considers the runtime one auto-installed and --auto-remove would take it
# with the rest. Rather than hardcode a package name that tracks the soname
# (libfst26 today), every shared library the built module actually links is
# looked up and marked manual. Nothing to update when OpenFST bumps.
#
# readlink -f matters: ldd reports /lib/..., dpkg records /usr/lib/... under
# usrmerge, so querying ldd's path directly matches nothing and the loop would
# quietly keep nothing at all. The counter turns that into a build failure, and
# the import at the end is the backstop -- it loads _fst, so a missing runtime
# library fails the build rather than shipping a broken image.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        netcat-traditional \
        python3 \
        python3-pip \
    && apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        libfst-dev \
        python3-dev \
    # No "pip install -U setuptools wheel" here: the vendored library is a
    # PEP 517 build, so pip creates an isolated build environment and fetches
    # the backend named in lib/pyproject.toml itself -- the system copies are
    # never used to build anything. Upgrading them also breaks outright on
    # Debian releases that ship python3-wheel from apt, since that has no RECORD
    # file for pip to uninstall ("Cannot uninstall wheel ... no RECORD file").
    && pip3 install --no-cache-dir -r requirements.txt \
    && fst_so="$(python3 -c 'import glob, pathlib, speech_to_phrase as s; print(glob.glob(str(pathlib.Path(s.__file__).parent / "_fst*.so"))[0])')" \
    && ldd "$fst_so" \
    && kept=0 \
    && for lib in $(ldd "$fst_so" | awk '/=> \// {print $3}'); do \
           pkg="$(dpkg -S "$(readlink -f "$lib")" 2>/dev/null | cut -d: -f1 | head -1)"; \
           if [ -n "$pkg" ]; then \
               echo "keeping $pkg (provides $lib)"; \
               apt-mark manual "$pkg"; \
               kept=$((kept + 1)); \
           fi; \
       done \
    && if [ "$kept" -eq 0 ]; then \
           echo "ERROR: mapped none of _fst's shared libraries to packages; refusing to purge" >&2; \
           exit 1; \
       fi \
    && apt-get purge -y --auto-remove cmake g++ libfst-dev python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache \
    && python3 -c "import speech_to_phrase; print('vendored speech_to_phrase imports OK after purge')"

# Bundle the default acoustic model so a fresh install works offline and doesn't
# spend its first boot downloading ~140 MB. Only src/models.py is copied first,
# so editing the rest of the app doesn't invalidate this layer and re-download.
# Build with --build-arg BUNDLE_MODEL= for a lean image that fetches on demand.
ARG BUNDLE_MODEL=stt_en_citrinet_512
COPY src/models.py ./src/models.py
RUN \
    if [ -n "${BUNDLE_MODEL}" ]; then \
        python3 -c "import sys; sys.path.insert(0, 'src'); import models; from pathlib import Path; \
                    models.ensure_model('${BUNDLE_MODEL}', Path('/usr/src/models'))" \
        && du -sh "/usr/src/models/${BUNDLE_MODEL}"; \
    fi

# Add-on application + intent sentence patches. The complete sentence/response
# data comes from home-assistant-intents; sentences/ only overrides individual
# slot combinations until those fixes reach a package release. config.yaml is
# read at runtime for the version advertised over Wyoming.
COPY src/ ./src/
COPY sentences/ ./sentences/
COPY example_values.yaml config.yaml ./

WORKDIR /
COPY rootfs /

# The Wyoming server answers a describe with the add-on name.
HEALTHCHECK --start-period=15m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10300 \
        | grep -iq "speech-to-phrase" \
        || exit 1
