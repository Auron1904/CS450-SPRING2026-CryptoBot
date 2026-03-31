# Use the high-speed uv image to build the environment
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set the working directory inside the container
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 1. Copy only the environment files first (for fast caching)
COPY pyproject.toml uv.lock ./

# 2. Install dependencies (frozen follows your lockfile exactly)
RUN uv sync --frozen --no-install-project --no-dev

# 3. Copy your project folders (src, data, etc.) and app.py
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .

# 4. Final sync to include the project code
RUN uv sync --frozen --no-dev

# -- Runtime Stage (The actual small container that runs) --
FROM python:3.12-slim-bookworm
WORKDIR /app

# Copy the prepared environment from the builder stage
COPY --from=builder /app /app

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Open the port for Streamlit
EXPOSE 8501

# Command to launch your dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]