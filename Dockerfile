# Use a slim Python image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project configuration and source code
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY alphakhulnasoft/ ./alphakhulnasoft/

# Install dependencies (using uv for speed and precision)
# Remove --frozen flag to allow resolution, and --no-cache to reduce image size
RUN uv sync --no-dev

# Copy additional files
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-m", "alphakhulnasoft.benchmark"]
