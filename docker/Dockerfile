# Use the official lightweight Python image
FROM python:3.12.3-slim

# Install poetry
RUN python -m pip install poetry==1.8.4

# Set the working directory
WORKDIR /

# Copy only the dependency files first (for caching purposes)
COPY poetry.lock ./
COPY pyproject.toml ./

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev --without test

# Copy the application code
COPY ./src ./src

# Ensure directories exist (they will be mounted)
RUN mkdir -p output data logs models config

# Set the entrypoint command
ENTRYPOINT ["poetry", "run", "python", "src/main.py"]
