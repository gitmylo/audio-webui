# STAGE 1: Setup Python environment and clone the repository
FROM python:3.10-slim AS deps

# Install build dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends git g++ build-essential

# Upgrade pip and numpy
RUN pip install --upgrade pip numpy

FROM deps AS builder

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the application
RUN python install.py --skip-venv

# STAGE 2: Build final image
FROM python:3.10-slim

LABEL org.opencontainers.image.url="https://github.com/gitmylo/audio-webui/"
LABEL org.opencontainers.image.authors="gitmylo"
LABEL org.opencontainers.image.name="audio-webui"
LABEL org.opencontainers.image.description="A webui for different audio related Neural Networks "

# Copy python dependencies from setup to /usr/local
COPY --from=builder /usr/local /usr/local

# Set the working directory in the container to /app
WORKDIR /app

# Copy the application from setup
COPY --from=builder /app .

# Set environment variables for username and password
ENV USERNAME user
ENV PASSWORD password

# Expose port 8000
EXPOSE 8000

# Define command to run the app using the environment variables for username and password
CMD ["sh", "-c", "python main.py -sv -si -s -u ${USERNAME} -p ${PASSWORD}"]
