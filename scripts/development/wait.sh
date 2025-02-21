#!/bin/bash

echo "Seeding .bashrc with Google Cloud credentials..."
touch ~/.bashrc
mkdir -p /root/.google
if ! grep -qxF "alias awslocal='awslocal --endpoint=\${LOCALSTACK_ENDPOINT}'" ~/.bashrc; then
  echo "alias awslocal='awslocal --endpoint=\${LOCALSTACK_ENDPOINT}'" > ~/.bashrc
  echo "alias cleandeps='cd /app/bili-core/scripts/development && CLEAN=true ./install-deps'" >> ~/.bashrc
  echo "alias createpgdb='psql \"\${POSTGRES_CONNECTION_STRING%/*}/postgres\" -c \"CREATE DATABASE langgraph;\" || true'" >> ~/.bashrc
  echo "alias deps='cd /app/bili-core/scripts/development && ./install-deps'" >> ~/.bashrc
  echo "alias seeds3='cd /app/bili-core/scripts/development && ./upload-to-localstack-s3'" >> ~/.bashrc
  echo "alias streamlit='createpgdb && deps && cd /app/bili-core/scripts/development && ./start-streamlit-server'" >> ~/.bashrc
  echo "alias flask='createpgdb && deps && cd /app/bili-core/scripts/development && ./start-flask-server'" >> ~/.bashrc
  echo "export LOCALSTACK_ENDPOINT='http://bili-core-localstack:4566'" >> ~/.bashrc
  echo '[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] && gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"' >> ~/.bashrc
  echo '[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] && gcloud config set project "$GOOGLE_PROJECT_ID"' >> ~/.bashrc
  echo 'source /app/bili-core/scripts/development/secrets' >>~/.bashrc
fi
source ~/.bashrc

# Update /etc/hosts with the LocalStack hostname for OpenSearch DNS resolution
echo "Updating /etc/hosts with LocalStack hostname for OpenSearch DNS resolution..."
LOCALSTACK_HOSTNAME="bili-core-localstack"
LOCALSTACK_IP=$(getent hosts $LOCALSTACK_HOSTNAME | awk '{ print $1 }')

if [ -n "$LOCALSTACK_IP" ]; then
  echo "$LOCALSTACK_IP bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud" >> /etc/hosts
  echo "Updated /etc/hosts with LocalStack IP: $LOCALSTACK_IP"
else
  echo "Failed to retrieve LocalStack IP address."
fi

# Wait for s3 bucket to be created
echo "Waiting for LocalStack S3 bucket to be created..."
until awslocal --endpoint-url=http://bili-core-localstack:4566 s3 ls; do
  echo "LocalStack S3 bucket not found. Retrying..."
  sleep 1
done

# Upload files from data directory to Localstack s3 bucket
# The community edition does not persist s3 files between container restarts
echo "Uploading files to LocalStack S3 bucket..."
mkdir -p /app/bili-core/data
/app/bili-core/scripts/development/upload-to-localstack-s3 /app/bili-core/data

echo "Bili container is ready."
# Infinite wait so that we can attach to container
trap : TERM INT
tail -f /dev/null &
wait
