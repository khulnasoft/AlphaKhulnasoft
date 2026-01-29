# Use a slim Python image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies (using uv for speed and precision)
RUN uv sync --frozen --no-cache

# Copy the rest of the application
COPY alphakhulnasoft/ ./alphakhulnasoft/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-m", "alphakhulnasoft.benchmark"]
