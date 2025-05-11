@echo off

:: Use GPU profile if NVIDIA GPU is detected, otherwise use CPU profile
:: Note: nvidia-smi must be in the PATH
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Using GPU profile.
    set profile=gpu
) else (
    echo No NVIDIA GPU detected. Using CPU profile.
    set profile=cpu
)

:: Change to the directory of this script
cd %~dp0

:: Build the Docker images
docker build ..\.. -t bili-core
docker-compose build

:: Create the network if it doesn't already exist
docker network create bili-core 2>nul

:: Start the containers with the appropriate profile
docker-compose --profile %profile% up

:: Stop the running containers
docker stop bili-core
docker stop bili-core-postgis
docker stop bili-core-mongodb
docker stop bili-core-localstack

:: Bring down the Docker Compose setup
docker-compose down

:end
echo Done
