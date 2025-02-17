#!/bin/bash

set -e
set -x

# Clean up any previous package directory
rm -rf dist/

# Create a directory for the package
mkdir -p dist

# Install dependencies
pip install -r requirements.txt -t dist/

# Retrieve AWS certificate pem file for secure connections
wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem -O dist/global-bundle.pem

# Copy the bili code into the dist directory
# Don't copy any files containing the string "test", __pycache__, .pyc, .git, and .DS_Store files
rsync -av --exclude='*test*' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='.git' \
      --exclude='.DS_Store' \
      "bili/" dist/bili/

# Copy the .streamlit directory into the dist directory
rsync -av ".streamlit/" dist/.streamlit/
