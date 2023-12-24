# Use an official Python runtime as a parent image
FROM python:3.10.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add metadata to the image to describe that the container is used for
LABEL description="Python Development Container"

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
