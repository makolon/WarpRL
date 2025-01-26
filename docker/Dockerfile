FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG WARPRL_PATH_ARG
ARG DOCKER_USER_HOME_ARG
ENV WARPRL_PATH=${WARPRL_PATH_ARG}
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}

# Set up user
WORKDIR ${WARPRL_PATH}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-c"]

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    freeglut3-dev \
    gcc \
    git \
    ibverbs-providers \
    libffi-dev \
    libglfw3 \
    libglfw3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglew-dev \
    libglu1-mesa-dev \
    libibverbs1 \
    libjpeg-dev \
    libosmesa6-dev \
    libpng-dev \
    librdmacm1 \
    libssl-dev \
    libvulkan1 \
    libx11-6 \
    mesa-utils \
    openjdk-8-jdk \
    openssh-client \
    patchelf \
    swig \
    unzip \
    vim \
    wget \
    x11-xserver-utils \
    xvfb \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="${DOCKER_USER_HOME}/.cargo/bin/:$PATH"

# Copy the files so that Python packages can be installed using uv
COPY ../pyproject.toml ./
COPY ../uv.lock ./

# Run uv sync
RUN . ${DOCKER_USER_HOME}/.bashrc
