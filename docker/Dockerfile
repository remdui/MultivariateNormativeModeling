# Use the official Python image from the Docker Hub
FROM python:3.12.3-slim AS builder

# Install any needed packages specified in pyproject.toml
RUN pip install poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory in the container
WORKDIR /

COPY pyproject.toml ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root


FROM python:3.12.3-slim AS runtime

ENV VIRTUAL_ENV=/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the current directory contents into the container at /app
COPY /config ./
COPY /src ./
COPY /data ./
COPY /tests ./

# Run main.py when the container launches
CMD ["poetry", "run", "python", "src/main.py"]