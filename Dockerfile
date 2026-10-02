# Quadrants on ROCm 10.0.0.
#
# Build from the root of this repository:
#
#   docker build -t quadrants-release-rocm10 .
#
# The build clones this release branch and its submodules. Override that
# checkout with:
#
#   docker build \
#     --build-arg QUADRANTS_REPO=https://github.com/AMD-Ecosystem/quadrants.git \
#     --build-arg QUADRANTS_REF=rocm10.0.0_r26.10 \
#     -t quadrants-release-rocm10 .
#
# Run on a host whose kernel driver matches ROCm 10, with the GPU devices passed in:
#
#   docker run --rm -it \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add "$(stat -c '%g' /dev/kfd)" \
#     -e HIP_VISIBLE_DEVICES=0 \
#     -e ROCR_VISIBLE_DEVICES=0 \
#     -e CUDA_VISIBLE_DEVICES=0 \
#     quadrants-release-rocm10
#
# The base image supplies the ROCm 10 userspace (/opt/rocm). Quadrants is compiled
# in the build stage with C++ tests on. The final image keeps the virtualenv,
# quadrants_cpp_tests, and this Quadrants tree. It does not install Genesis.

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.0.0-full@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374

FROM ${ROCM_IMAGE} AS build

ARG QUADRANTS_REPO=https://github.com/AMD-Ecosystem/quadrants.git
ARG QUADRANTS_REF=rocm10.0.0_r26.10

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        build-essential \
        clang \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        xz-utils \
        lsb-release \
        liblz4-dev \
        libssl-dev \
        libncurses-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && pip install -U "pip>=25.1"

# Clone the release branch, then initialize its submodules. ./build.py
# downloads LLVM 22 and the Vulkan SDK, then builds the wheel. AMDGPU is off
# by default; CUDA is on by default and is turned off because this image has
# no CUDA toolkit. Vulkan stays off for a ROCm-only runtime. QD_BUILD_TESTS
# builds quadrants_cpp_tests.
RUN git clone --branch "${QUADRANTS_REF}" --single-branch "${QUADRANTS_REPO}" /src/quadrants

WORKDIR /src/quadrants
RUN git submodule update --init --recursive \
    && pip install --group dev \
    && CMAKE_ARGS="-DQD_WITH_AMDGPU:BOOL=ON -DQD_WITH_CUDA:BOOL=OFF -DQD_WITH_VULKAN:BOOL=OFF -DQD_BUILD_TESTS:BOOL=ON" \
       ./build.py wheel \
    && mkdir -p /opt/quadrants \
    && find /src/quadrants/build -name quadrants_cpp_tests -type f -exec cp {} /opt/quadrants/quadrants_cpp_tests \; \
    && test -x /opt/quadrants/quadrants_cpp_tests

RUN pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "torch[device-all]==2.13.0+rocm10.0.0" \
    && pip install /src/quadrants/dist/quadrants-*.whl \
    && pip install --group test \
    && rm -rf /src/quadrants/build /src/quadrants/dist


FROM ${ROCM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib:/opt/rocm/lib \
    ROCM_PATH=/opt/rocm \
    PYTHONPATH=/opt/rocm/share/amd_smi \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        libpython3.12 \
        libgomp1 \
        ca-certificates \
        git \
        gcc \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/quadrants/quadrants_cpp_tests /opt/quadrants/quadrants_cpp_tests
COPY --from=build /src/quadrants /src/quadrants

# Quadrants dlopens "libamdhip64.so". Point that name at the ROCm runtime
# shipped inside the PyTorch wheel so the process does not load a second HIP.
RUN ln -sf libamdhip64.so.7 \
    /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so

WORKDIR /src/quadrants

CMD ["python", "-c", "import torch, quadrants; print('torch', torch.__version__); print('quadrants', quadrants.__version_str__)"]
