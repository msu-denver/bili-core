FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG TARGETARCH
RUN echo "Building for $TARGETARCH"

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app/bili-core

# Install common installation dependencies
RUN apt-get update && apt-get install -yq apt-transport-https ca-certificates wget dirmngr gnupg software-properties-common

RUN apt-get update && apt-get install -yq \
    build-essential \
    cmake \
    curl \
    default-jdk \
    dos2unix \
    git \
    git-core \
    less \
    libffi-dev \
    libhdf5-dev \
    libpq-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libz-dev \
    lsof \
    nasm \
    pkg-config \
    poppler-utils \
    unzip \
    vim \
    wget \
    xclip \
    xsel \
    xvfb

# Install Python 3.11 using deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -yq \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pytest
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install and upgrade pip using official script
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py

# Ensure the latest versions of pip, setuptools, and wheel are installed
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --upgrade setuptools==70.0.0

# Install AWS CLI
RUN if [ "$TARGETARCH" = "amd64" ]; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    else \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    fi && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Install LocalStack for local development and testing using AWS
RUN pip3 install localstack awscli-local boto3

# Install gcloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Set up repo for PostGIS client
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ jammy-pgdg main" | tee /etc/apt/sources.list.d/pgdg.list
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get update && apt-get install -y postgresql

# Set up repo for MongoDB client
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
       gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
       --dearmor
RUN echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
    | tee /etc/apt/sources.list.d/mongodb-org-7.0.list
RUN apt-get update && apt-get install -y mongodb-org

# Copy wait script and make it executable
COPY scripts/development/wait.sh /app/wait.sh
RUN dos2unix /app/wait.sh && chmod a+x /app/wait.sh

ENTRYPOINT ["/app/wait.sh"]
