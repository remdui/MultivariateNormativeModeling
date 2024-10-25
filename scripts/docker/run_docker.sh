#!/bin/bash

# Build the Docker image
docker-compose -f docker/docker-compose.yml build

# Run the Docker container with all provided arguments
docker-compose -f docker/docker-compose.yml run app "$@"
