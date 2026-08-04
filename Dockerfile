# syntax=docker/dockerfile:1
#
# Stable, with CUDA-accelerated PyTorch for the deep trackers (CoTracker3, TAPIR).
#
# You do not need this image for CPU-only use -- `pip install stable-tracker`
# on any machine gives you `lk`/`template` plus automatic CPU fallback for the
# deep trackers. Build this to run the GPU path in a container, e.g. on a
# headless Linux box with an NVIDIA card. On a Windows workstation, running
# natively is simpler and lets `stable serve` open your browser for you -- see
# docs/windows-gpu-setup.md.
#
# Build:
#   docker build -t stable-tracker:cuda .
#
# Run (needs the NVIDIA Container Toolkit on the host; --gpus all requires
# Docker >= 19.03). Workspace and model cache are named volumes so projects
# and downloaded checkpoints survive a container restart:
#   docker run --rm -it --gpus all -p 8000:8000 \
#     -v stable-workspace:/root/.stable/projects \
#     -v stable-models:/root/.stable/models \
#     stable-tracker:cuda
#
# Then open http://localhost:8000 -- the container has no display, so
# `stable serve` runs with --no-browser; open the URL yourself.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# PyTorch's published wheels track specific CUDA minor versions; pin the wheel
# index to match this image's runtime rather than whatever a plain `pip
# install torch` happens to resolve. Override at build time (--build-arg) to
# track a different CUDA/PyTorch pairing.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STABLE_WORKSPACE_DIR=/root/.stable/projects \
    STABLE_MODEL_DIR=/root/.stable/models

# Python 3.11 (Ubuntu 22.04 ships 3.10) via deadsnakes, ffmpeg for stable.video,
# git so pip can install the trackers' model-code packages straight from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl git ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade pip

WORKDIR /app
# Only what the package build needs -- see .dockerignore for what is excluded
# (web/ source, tests/, node_modules, etc.). The frontend is already built
# into stable/server/webdist (scripts/build_frontend.py), so this image never
# needs Node.js.
COPY pyproject.toml README.md LICENSE ./
COPY stable ./stable

RUN python3.11 -m pip install ".[video,server]" \
    && python3.11 -m pip install torch --index-url "${TORCH_INDEX_URL}" \
    && python3.11 -m pip install \
        "tapnet @ git+https://github.com/google-deepmind/tapnet.git" \
        "cotracker @ git+https://github.com/facebookresearch/co-tracker.git"

EXPOSE 8000
VOLUME ["/root/.stable"]

ENTRYPOINT ["stable"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--no-browser", "--no-build"]
