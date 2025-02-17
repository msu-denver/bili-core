@echo off
:: https://docs.localstack.cloud/user-guide/aws/opensearch/#opensearch-dashboards

:: Exit on error
setlocal enabledelayedexpansion
set EXIT_ON_ERROR=0

:: Change to the directory of the script
cd /d "%~dp0"

:: Wait for bili-core-localstack container to have started
echo Waiting for LocalStack container to have started...
:wait_for_localstack
for /f "delims=" %%R in ('docker inspect -f "{{.State.Running}}" bili-core-localstack 2^>nul') do (
    set "IS_RUNNING=%%R"
)
if "!IS_RUNNING!" neq "true" (
    timeout /t 5 >nul
    echo Waiting for LocalStack container to have started...
    goto wait_for_localstack
)

:: Get LocalStack container's IP address
for /f "delims=" %%A in ('docker inspect bili-core-localstack ^| jq -r ".[0].NetworkSettings.Networks | to_entries | .[].value.IPAddress"') do (
    set "LOCALSTACK_IP=%%A"
)
echo LocalStack IP address: %LOCALSTACK_IP%

:: Start OpenSearch Dashboards container
:: The version of the Dashboard has to exactly match the version of OpenSearch that LocalStack is running
echo Starting OpenSearch Dashboards container...
docker run --rm --name bili-core-opensearch-dashboard ^
  --network bili-core ^
  --dns "%LOCALSTACK_IP%" ^
  -p 5601:5601 ^
  -e "OPENSEARCH_HOSTS=http://bilicore-dev.us-east-1.opensearch.localhost.localstack.cloud:4566" ^
  -e "DISABLE_SECURITY_DASHBOARDS_PLUGIN=true" ^
  opensearchproject/opensearch-dashboards:2.11.1

:: Exit the script
endlocal
exit /b %EXIT_ON_ERROR%
