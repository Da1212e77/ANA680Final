# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Copy local code to the container image.
WORKDIR /app
COPY . .

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
