# EPANSA Orchestrator

Multi-tenant knowledge graph system for personal data synchronization with Google services (Calendar, Contacts, Drive, Photos).

## Overview

EPANSA Orchestrator is a FastAPI-based backend system that:
- Authenticates users via Google OAuth
- Synchronizes data from Google services
- Builds and maintains personal knowledge graphs in Neo4j
- Provides secure storage for OAuth tokens in PostgreSQL
- Uses AI/ML models for data processing and embeddings

## Tech Stack

- **Python 3.11+** - Core language
- **FastAPI** - Web framework
- **Neo4j 5** - Graph database for knowledge graphs
- **PostgreSQL 15** - Relational database for secure token storage
- **Docker** - Containerization
- **uv** - Fast Python package installer and resolver

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Docker and Docker Compose
- Python 3.11 or higher (if running locally)

## How to setup client_secret.json for Google OAuth
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one for Web application.
3. Navigate to "APIs & Services" > "Credentials".
4. Click "Create Credentials" and select "OAuth 2.0 Client IDs".
5. Choose "Web application" as the application type.
6. Set the authorized redirect URIs to `http://localhost:5000/auth/callback
7. Download the `client_secret.json` file.
8. Place the `client_secret.json` file in the project root directory.

### Installing uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Local Development Setup

### 1. Install Dependencies

```bash
# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp example.env .env

# Edit .env with your actual configuration
# - Database credentials
# - Neo4j credentials
# - Google OAuth client credentials
# - JWT secrets
```

### 3. Run with Docker Compose

```bash
# Start all services (PostgreSQL, Neo4j, API)
docker compose up -d

# View logs
docker compose logs -f api

# Stop all services
docker compose down
```

The API will be available at `http://localhost:5000`
Neo4j Browser will be available at `http://localhost:7474`

## Package Management with uv

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add a dependency with version constraint
uv add "package-name>=1.0.0,<2.0.0"
```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update a specific package
uv add package-name --upgrade
```

### Removing Dependencies

```bash
# Remove a dependency
uv remove package-name
```

## Docker Build

The Dockerfile uses a multi-stage build with uv:

1. **Builder stage**: Installs dependencies using uv into a virtual environment
2. **Runtime stage**: Copies the virtual environment and application code

Benefits:
- Smaller final image size
- Faster builds with better caching
- Reproducible builds with `uv.lock`

## Project Structure

```
.
├── backend/              # FastAPI application
│   ├── app.py           # Main application entry point
│   ├── core/            # Core utilities (config, auth, security)
│   ├── models/          # Pydantic models
│   ├── routers/         # API route handlers
│   └── services/        # Business logic services
├── libs/                # Shared libraries
│   └── llm_graph_builder/  # Knowledge graph builder
├── data/                # Runtime data (sync state, temp files)
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # Locked dependency versions
├── Dockerfile           # Multi-stage Docker build
└── docker-compose.yml   # Service orchestration
```

## API Endpoints

- `POST /auth/login` - Initiate Google OAuth flow
- `POST /auth/callback` - OAuth callback handler
- `GET /auth/me` - Get current user info
- `POST /sync/trigger` - Trigger manual sync for a user
- More endpoints documented in the FastAPI auto-generated docs at `/docs`

## Development Workflow

```bash
# Install dependencies
uv sync

# Run the API locally (without Docker)
uvicorn backend.app:app --reload --port 5000

# Run tests (if available)
uv run pytest

# Format code (if you add formatters)
uv run black backend/
uv run isort backend/

# Type checking (if you add mypy)
uv run mypy backend/
```

## Migration from requirements.txt

This project has been migrated from `requirements.txt` to `pyproject.toml` with `uv`:

### Old workflow:
```bash
pip install -r requirements.txt
pip install -r libs/llm_graph_builder/requirements.txt
```

### New workflow:
```bash
uv sync
```

All dependencies from both `requirements.txt` files are now consolidated in `pyproject.toml`.

## Troubleshooting

### Docker build fails
- Ensure you have the latest version of uv in the Dockerfile
- Clear Docker build cache: `docker builder prune`

### Dependencies not installing
- Delete `.venv` and run `uv sync` again
- Check `uv.lock` is committed to version control

### Module not found errors
- Ensure you've activated the virtual environment
- Run `uv sync` to ensure all dependencies are installed

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
