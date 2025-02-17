#!/bin/bash

# Function to seed LocalStack
# Manage LocalStack by downloading LocalStack Desktop at https://app.localstack.cloud/download
seed_localstack() {
  echo "Waiting for LocalStack to be ready..."
  until awslocal ssm describe-parameters >/dev/null 2>&1; do
    sleep 5
    echo "Waiting for LocalStack to be ready..."
  done

  # https://docs.localstack.cloud/user-guide/aws/ssm/
  # List parameters to confirm seeding:
  # awslocal ssm describe-parameters --endpoint-url=http://bili-core-localstack:4566
  # Fetch a specific parameter:
  # awslocal ssm get-parameter --name "/local/myapp/app_key" --with-decryption --endpoint-url=http://bili-core-localstack:4566
  echo "Seeding Parameter Store..."
  # An example of seeding Parameter Store with test values using inline JSON
  #awslocal ssm put-parameter --cli-input-json '{
  #  "Name": "/local/myapp/app_key",
  #  "Value": "12345ABCDEFG",
  #  "Type": "SecureString"
  #}'
  echo "Parameter Store seeded successfully."

  # https://docs.localstack.cloud/user-guide/aws/ses/
  # List verified email identities:
  # awslocal ses list-identities --identity-type EmailAddress --endpoint-url=http://bili-core-localstack:4566
  # Check verification status of the email identity:
  # awslocal ses get-identity-verification-attributes --identities noreply@bilicore.dev --endpoint-url=http://bili-core-localstack:4566
  echo "Creating a local SES email identity..."
  awslocal ses verify-email-identity --email noreply@bilicore.dev
  echo "Local SES email identity created successfully."

  # https://docs.localstack.cloud/user-guide/aws/s3/
  # List all S3 buckets:
  # awslocal s3api list-buckets --endpoint-url=http://bili-core-localstack:4566
  # Check the bucket’s configuration:
  # awslocal s3api get-bucket-location --bucket bilicore-dev --endpoint-url=http://bili-core-localstack:4566
  # Upload a test file to the bucket:
  # awslocal s3 cp test-file.txt s3://bilicore-dev/test-file.txt --endpoint-url=http://bili-core-localstack:4566
  # Verify the test file upload:
  # awslocal s3 ls s3://bilicore-dev/ --endpoint-url=http://bili-core-localstack:4566
  echo "Creating a local S3 bucket..."
  awslocal s3api create-bucket --bucket bilicore-dev
  echo "Local S3 bucket created successfully."

  # https://docs.localstack.cloud/user-guide/aws/opensearch/
  # Check cluster health:
  # docker exec -it bili-core-localstack \
  #    curl -s http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/_cluster/health | jq
  # List all indexes to confirm creation:
  # docker exec -it bili-core-localstack \
  #   curl -X GET http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/_cat/indices?v
  # Check a specific index’s health (replace <index_name> with an actual index name):
  # docker exec -it bili-core-localstack \
  #   curl -X GET http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/<index_name> | jq
  echo "Creating a local OpenSearch cluster..."
  # Create the cluster
  awslocal opensearch create-domain --domain-name bilicore-dev
  # Wait for the cluster to be created
  while true; do
    CLUSTER_STATUS=$(curl -s http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/_cluster/health |
      grep -o '"status":"[^"]*"' |
      awk -F':' '{print $2}' |
      tr -d '"')

    if [ "$CLUSTER_STATUS" = "green" ] || [ "$CLUSTER_STATUS" = "yellow" ]; then
      echo "OpenSearch cluster is ready (status: $CLUSTER_STATUS)"
      break
    else
      echo "Waiting for OpenSearch cluster to be ready (current status: $CLUSTER_STATUS)..."
      sleep 5
    fi
  done
  # Disable watermark checking in OpenSearch for local development
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/_cluster/settings" \
    -H "Content-Type: application/json" \
    -d '{
          "persistent": {
             "cluster.routing.allocation.disk.threshold_enabled": false
          }
        }'

  # Create and configure indexes
  # https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
  # https://aws.amazon.com/blogs/machine-learning/get-started-with-amazon-titan-text-embeddings-v2-a-new-state-of-the-art-embeddings-model-on-amazon-bedrock/
  # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions#embeddings
  echo "Creating Google Vertex AI Index (768-dim)..."
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/vertex_text-embedding-005" \
    -H "Content-Type: application/json" \
    -d '{
             "settings": {
               "index": {
                 "knn": true,
                 "number_of_replicas": 0
               }
             },
             "mappings": {
               "properties": {
                 "text": {
                   "type": "text"
                 },
                 "metadata": {
                   "properties": {
                     "page": {
                       "type": "long"
                     },
                     "source": {
                       "type": "text",
                       "fields": {
                         "keyword": {
                           "type": "keyword",
                           "ignore_above": 256
                         }
                       }
                     }
                   }
                 },
                 "vector_field": {
                   "type": "knn_vector",
                   "dimension": 768,
                   "method": {
                     "name": "hnsw",
                     "space_type": "l2",
                     "engine": "nmslib",
                     "parameters": {
                       "ef_construction": 512,
                       "m": 16
                     }
                   }
                 }
               }
             }
           }'

  # Reset read-only status (in case the index was set to read-only)
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/vertex_text-embedding-005/_settings" \
    -H "Content-Type: application/json" \
    -d '{
           "index": {
             "blocks.read_only_allow_delete": null
           }
        }'

  echo "Creating Amazon Bedrock Index (1024-dim)..."
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/amazon_titan-embed-text-v2" \
    -H "Content-Type: application/json" \
    -d '{
             "settings": {
               "index": {
                 "knn": true,
                 "number_of_replicas": 0
               }
             },
             "mappings": {
               "properties": {
                 "text": {
                   "type": "text"
                 },
                 "metadata": {
                   "properties": {
                     "page": {
                       "type": "long"
                     },
                     "source": {
                       "type": "text",
                       "fields": {
                         "keyword": {
                           "type": "keyword",
                           "ignore_above": 256
                         }
                       }
                     }
                   }
                 },
                 "vector_field": {
                   "type": "knn_vector",
                   "dimension": 1024,
                   "method": {
                     "name": "hnsw",
                     "space_type": "innerproduct",
                     "engine": "nmslib",
                     "parameters": {
                       "ef_construction": 512,
                       "m": 16
                     }
                   }
                 }
               }
             }
           }'

  # Reset read-only status (in case the index was set to read-only)
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/amazon_titan-embed-text-v2/_settings" \
    -H "Content-Type: application/json" \
    -d '{
           "index": {
             "blocks.read_only_allow_delete": null
           }
        }'

  echo "Creating Azure OpenAI Index (1536-dim)..."
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/azure_text-embedding-3-large" \
    -H "Content-Type: application/json" \
    -d '{
             "settings": {
               "index": {
                 "knn": true,
                 "number_of_replicas": 0
               }
             },
             "mappings": {
               "properties": {
                 "text": {
                   "type": "text"
                 },
                 "metadata": {
                   "properties": {
                     "page": {
                       "type": "long"
                     },
                     "source": {
                       "type": "text",
                       "fields": {
                         "keyword": {
                           "type": "keyword",
                           "ignore_above": 256
                         }
                       }
                     }
                   }
                 },
                 "vector_field": {
                   "type": "knn_vector",
                   "dimension": 3072,
                   "method": {
                     "name": "hnsw",
                     "space_type": "cosinesimil",
                     "engine": "nmslib",
                     "parameters": {
                       "ef_construction": 512,
                       "m": 16
                     }
                   }
                 }
               }
             }
           }'

  # Reset read-only status (in case the index was set to read-only)
  curl -X PUT "http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566/azure_text-embedding-3-large/_settings" \
    -H "Content-Type: application/json" \
    -d '{
           "index": {
             "blocks.read_only_allow_delete": null
           }
        }'

  echo "Index creation complete!"

}

# Run the seed function in the background
seed_localstack &
