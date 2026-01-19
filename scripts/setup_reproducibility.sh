#!/bin/bash
# =============================================================================
# EpisTwin Reproducibility Setup Script
# =============================================================================
# This script validates and prepares the environment for reproducing
# the experiments described in the IJCAI 2026 paper.
#
# Usage:
#   ./scripts/setup_reproducibility.sh           # Full setup
#   ./scripts/setup_reproducibility.sh --check-only  # Validation only
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
CHECK_ONLY=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check-only    Only validate environment, don't create directories"
            echo "  --verbose, -v   Show detailed output"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Track validation results
ERRORS=0
WARNINGS=0

# =============================================================================
# SYSTEM REQUIREMENTS CHECK
# =============================================================================

log_section "Checking System Requirements"

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [[ $PYTHON_MAJOR -ge 3 && $PYTHON_MINOR -ge 11 ]]; then
            log_success "Python $PYTHON_VERSION (>= 3.11 required)"
        else
            log_error "Python $PYTHON_VERSION found, but >= 3.11 required"
            ((ERRORS++))
        fi
    else
        log_error "Python 3 not found"
        ((ERRORS++))
    fi
}

# Check Docker
check_docker() {
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        log_success "Docker $DOCKER_VERSION installed"
        
        # Check if Docker daemon is running
        if docker info &> /dev/null; then
            log_success "Docker daemon is running"
        else
            log_error "Docker daemon is not running"
            ((ERRORS++))
        fi
    else
        log_error "Docker not found"
        ((ERRORS++))
    fi
}

# Check Docker Compose
check_docker_compose() {
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        log_success "Docker Compose v$COMPOSE_VERSION installed"
    elif command -v docker-compose &> /dev/null; then
        log_warning "Legacy docker-compose found. Please upgrade to Docker Compose v2"
        ((WARNINGS++))
    else
        log_error "Docker Compose not found"
        ((ERRORS++))
    fi
}

# Check uv package manager
check_uv() {
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log_success "uv $UV_VERSION installed"
    else
        log_warning "uv package manager not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        ((WARNINGS++))
    fi
}

# Check Git submodules
check_submodules() {
    if [[ -f ".gitmodules" ]]; then
        if [[ -d "libs/llm_graph_builder/src" ]]; then
            log_success "Git submodules initialized (libs/llm_graph_builder)"
        else
            log_warning "Git submodules not initialized. Run: git submodule update --init --recursive"
            ((WARNINGS++))
        fi
    fi
}

check_python
check_docker
check_docker_compose
check_uv
check_submodules

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

log_section "Checking Directory Structure"

REQUIRED_DIRS=(
    "data/tmp"
    "data/sync"
    "emb_model_cache"
    "emb_model_cache/cache_hf"
    "config"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        log_success "Directory exists: $dir"
    else
        if [[ "$CHECK_ONLY" == true ]]; then
            log_warning "Directory missing: $dir"
            ((WARNINGS++))
        else
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        fi
    fi
done

# =============================================================================
# CONFIGURATION FILES
# =============================================================================

log_section "Checking Configuration Files"

REQUIRED_FILES=(
    "example.env"
    "docker-compose.yml"
    "Dockerfile"
    "pyproject.toml"
)

OPTIONAL_FILES=(
    ".env"
    "client_secret.json"
    "config/hyperparameters.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        log_success "Required file exists: $file"
    else
        log_error "Required file missing: $file"
        ((ERRORS++))
    fi
done

for file in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        log_success "Optional file exists: $file"
    else
        log_warning "Optional file missing: $file"
        ((WARNINGS++))
    fi
done

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

log_section "Checking Environment Configuration"

if [[ -f ".env" ]]; then
    # Check required environment variables
    REQUIRED_VARS=(
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "NEO4J_URI"
        "NEO4J_USER"
        "NEO4J_PASSWORD"
        "ENCRYPTION_KEY"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env 2>/dev/null; then
            VALUE=$(grep "^${var}=" .env | cut -d'=' -f2-)
            if [[ -n "$VALUE" && "$VALUE" != "..." ]]; then
                log_success "Environment variable set: $var"
            else
                log_warning "Environment variable not configured: $var"
                ((WARNINGS++))
            fi
        else
            log_warning "Environment variable missing: $var"
            ((WARNINGS++))
        fi
    done
    
    # Check API keys
    API_KEYS=(
        "GROQ_API_KEY"
        "GEMINI_API_KEY"
    )
    
    log_info "Checking API keys..."
    for key in "${API_KEYS[@]}"; do
        if grep -q "^${key}=" .env 2>/dev/null; then
            VALUE=$(grep "^${key}=" .env | cut -d'=' -f2-)
            if [[ -n "$VALUE" && "$VALUE" != "..." ]]; then
                log_success "API key configured: $key"
            else
                log_warning "API key not set: $key (required for full functionality)"
                ((WARNINGS++))
            fi
        else
            log_warning "API key missing: $key"
            ((WARNINGS++))
        fi
    done
else
    log_warning ".env file not found. Copy from example.env: cp example.env .env"
    ((WARNINGS++))
fi

# =============================================================================
# DEPENDENCIES
# =============================================================================

log_section "Checking Dependencies"

if [[ -f "uv.lock" ]]; then
    log_success "Dependency lock file exists: uv.lock"
fi

if [[ -f "requirements.txt" ]]; then
    log_success "Requirements file exists: requirements.txt"
fi

if [[ -f "libs/llm_graph_builder/requirements.txt" ]]; then
    log_success "LLM Graph Builder requirements exist"
fi

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    log_success "Virtual environment exists: .venv"
else
    if [[ "$CHECK_ONLY" == true ]]; then
        log_warning "Virtual environment not found. Run: uv sync"
        ((WARNINGS++))
    else
        log_info "Creating virtual environment..."
        if command -v uv &> /dev/null; then
            uv sync
            log_success "Virtual environment created and dependencies installed"
        else
            log_warning "uv not found. Install dependencies manually: pip install -r requirements.txt"
            ((WARNINGS++))
        fi
    fi
fi

# =============================================================================
# SUMMARY
# =============================================================================

log_section "Summary"

echo ""
if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    log_success "All checks passed! Environment is ready for reproducibility."
elif [[ $ERRORS -eq 0 ]]; then
    log_warning "$WARNINGS warning(s) found. Environment may work but review warnings above."
else
    log_error "$ERRORS error(s) and $WARNINGS warning(s) found."
    log_error "Please fix errors before proceeding."
    exit 1
fi

echo ""
log_info "Next steps:"
echo "  1. Configure .env file with your API keys"
echo "  2. Place client_secret.json in project root (for Google OAuth)"
echo "  3. Start services: docker compose up -d"
echo "  4. Verify: docker compose ps"
echo ""

exit 0
