#!/bin/bash

# start_container.sh - Start NeMo 2.0 container for Japanese Continual Learning

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="nemo-japanese-cl"
IMAGE_NAME="nvcr.io/nvidia/nemo:25.04"
WORKSPACE_DIR=$(pwd)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NeMo 2.0 Japanese Continual Learning${NC}"
echo -e "${BLUE}Container Startup Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info 2>/dev/null | grep -q nvidia; then
    print_warning "NVIDIA Docker runtime not detected. GPU support may not be available."
fi

# Stop existing container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo -e "${BLUE}Stopping existing container...${NC}"
    docker stop "$CONTAINER_NAME" >/dev/null
    print_status "Stopped existing container"
fi

# Remove existing container if exists
if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
    echo -e "${BLUE}Removing existing container...${NC}"
    docker rm "$CONTAINER_NAME" >/dev/null
    print_status "Removed existing container"
fi

# Check if image exists, pull if not
if ! docker images -q "$IMAGE_NAME" | grep -q .; then
    echo -e "${BLUE}Pulling NeMo image...${NC}"
    docker pull "$IMAGE_NAME"
    print_status "Pulled NeMo image"
fi

# Load environment variables if .env exists
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE="--env-file .env"
    print_status "Found .env file, loading environment variables"
else
    print_warning "No .env file found. Copy .env.template to .env and configure it."
fi

# Determine container mode
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    CONTAINER_MODE="interactive"
    DOCKER_FLAGS="-it"
    print_status "Starting in interactive mode"
elif [ "$1" = "--daemon" ] || [ "$1" = "-d" ]; then
    CONTAINER_MODE="daemon"
    DOCKER_FLAGS="-d"
    print_status "Starting in daemon mode"
else
    CONTAINER_MODE="interactive"
    DOCKER_FLAGS="-it"
    print_status "Starting in interactive mode (default)"
fi

# Start the container
echo -e "${BLUE}Starting NeMo container...${NC}"

docker run $DOCKER_FLAGS \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    -v "$WORKSPACE_DIR:/workspace" \
    -w /workspace \
    $ENV_FILE \
    "$IMAGE_NAME" \
    ${2:-bash}

if [ $? -eq 0 ]; then
    if [ "$CONTAINER_MODE" = "daemon" ]; then
        print_status "Container started in daemon mode"
        echo -e "${BLUE}To attach to the container, run:${NC}"
        echo -e "  docker exec -it $CONTAINER_NAME bash"
        echo -e "${BLUE}To stop the container, run:${NC}"
        echo -e "  docker stop $CONTAINER_NAME"
    else
        print_status "Container session completed"
    fi
else
    print_error "Failed to start container"
    exit 1
fi

echo -e "${BLUE}========================================${NC}" 