# EpisTwin: A Neuro-Symbolic Framework for Personal Knowledge Graph Construction and Reasoning

[![IJCAI 2026](https://img.shields.io/badge/IJCAI-2026-blue.svg)](https://2026.ijcai.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper**: *EpisTwin: Neuro-Symbolic Personal Knowledge Graphs for Trustworthy Personal AI*  
> **Venue**: IJCAI 2026  
> **arXiv**: [arXiv link placeholder]

---

## Abstract

Personal Artificial Intelligence is currently constrained by the fragmentation of user data across isolated silos. While Retrieval-Augmented Generation (RAG) offers a partial remedy, its reliance on unstructured vector similarity fails to capture the latent semantic topology and temporal dependencies essential for holistic sensemaking.

**EpisTwin** is a neuro-symbolic framework that grounds generative reasoning in a verifiable, user-centric **Personal Knowledge Graph (PKG)**. The system uses Multimodal Language Models to populate a PKG where heterogeneous data from multiple applications are lifted into semantic triples. At inference time, Graph Retrieval-Augmented Generation enables complex reasoning over the semantic personal graph. The generation of grounded answers is controlled by an **agentic coordinator** that can apply **Visual-Symbolic Transduction**, effectively enabling contextual reasoning over images through a mechanism that dynamically re-grounds symbolic entities in their raw visual payload when epistemic uncertainty is high.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Algorithms](#algorithms)
  - [PKG Population Task (Ψ)](#pkg-population-task-ψ)
  - [Reasoning Engine](#reasoning-engine)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Model Configurations](#model-configurations)
  - [Hyperparameters](#hyperparameters)
- [Reproducing Experiments](#reproducing-experiments)
- [Dataset and Evaluation](#dataset-and-evaluation)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Architecture Overview

<p align="center">
  <img src="images/PKG_aggregation.pdf" alt="PKG Population Architecture" width="100%"/>
</p>

**Figure 1**: PKG population when the Information Object is a photo: triples are extracted from both metadata and visual content.

<p align="center">
  <img src="images/communities_2.pdf" alt="Community Detection" width="100%"/>
</p>

**Figure 2**: Communities over the PKG: (a) Topologically disjoint entities grouped into shared communities reveal implicit consequentiality. (b) Macroscopic visualization of a PKG populated by entities, relationships, and thematic communities.

### System Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **PKG Population Engine** | Transforms Information Objects into semantic triples | `backend/services/pkg_population.py` |
| **GraphRAG Indexer** | Community detection and summarization | `backend/services/initialize_graphRag.py` |
| **Core Agent (Δ_Core)** | LangGraph-based reasoning orchestrator | `backend/services/conversation.py` |
| **Fallback Agent (Δ_FB)** | Neural co-routine for unstructured modalities | `backend/services/conversation.py` |
| **Visual Analyzer (t_VIS)** | Online Visual Refinement via VQA | `backend/services/analyzer.py` |
| **Image Captioner (τ)** | Visual-to-text transduction | `backend/services/img_description.py` |

### Data Flow

```
Google Services → Information Objects (σ, μ, c) → PKG Population (Φ_M ∪ Φ_C) → Neo4j PKG
                                                                                    ↓
User Query → Core Agent (Δ_Core) → GraphRAG Local Search → Response
                      ↓ (if v_t = Insufficient)
              Fallback Agent (Δ_FB) → Visual Refinement (t_VIS) → VQA Analysis
```

---

## Algorithms

### PKG Population Task (Ψ)

The PKG Population Task models the update function Ψ that transitions the graph from state G^k_u to G^(k+1)_u:

```
G^(k+1)_u = Ψ(G^k_u, ι^(k+1)) = G^k_u ⊕ (Φ_M(μ) ∪ Φ_C(c))
```

#### Algorithm 1: PKG Population

```
Algorithm: PKG_POPULATION(ι, G_u)
─────────────────────────────────────────────────────────────────
Input: Information Object ι = (σ, μ, c), Current PKG G_u
Output: Updated PKG G'_u

1:  // Phase 1: Metadata Triples Extraction (Φ_M)
2:  T_meta ← ∅
3:  for each (key, value) ∈ μ do
4:      predicate ← ρ(key)           // Map key to predicate
5:      object ← λ(value)            // Cast value to entity/literal
6:      T_meta ← T_meta ∪ {(n_ι, predicate, object)}
7:  end for

8:  // Phase 2: Unstructured Content Extraction (Φ_C)
9:  if c ∈ C_vis then                // Visual content
10:     ĉ ← τ(c)                     // Apply captioning operator
11: else
12:     ĉ ← c                        // Text remains unchanged
13: end if
14: T_content ← f_KGC(ĉ)             // LLM-based triple extraction

15: // Phase 3: Graph Merge (⊕ operator)
16: G_new ← BUILD_SUBGRAPH(T_meta ∪ T_content)
17: G'_u ← MERGE(G_u, G_new)         // Entity resolution + linking

18: // Phase 4: Community Detection (Post-processing)
19: P ← LEIDEN(G'_u)                 // Detect community structure
20: for each P_i ∈ P do
21:     n_P ← CREATE_COMMUNITY_NODE(P_i)
22:     S_P ← f_SUM(P_i, T_P)        // Generate summary via LLM
23:     ATTACH_SUMMARY(n_P, S_P)
24:     for each n ∈ P_i do
25:         ADD_EDGE(n, IN_COMMUNITY, n_P)
26:     end for
27: end for

28: return G'_u
─────────────────────────────────────────────────────────────────
```

**Implementation**: [`backend/services/pkg_population.py`](backend/services/pkg_population.py)

#### Visual Captioning Operator (τ)

The captioning operator τ transforms visual content into textual descriptions:

```
τ(c) ~ P_φ(· | c, prompt_vis)
```

**Implementation**: [`backend/services/img_description.py`](backend/services/img_description.py)

---

### Reasoning Engine

The reasoning process is modeled as a sequential decision-making problem with the **Core Agent** Δ_Core acting as the primary controller.

#### Algorithm 2: Agentic Reasoning Workflow

```
Algorithm: EPISTWIN_REASONING(q, G_u)
─────────────────────────────────────────────────────────────────
Input: User query q, Personal Knowledge Graph G_u
Output: Grounded response r

1:  // Initialize state
2:  s_0 ← (q, H_0 = ∅)               // H_t is reasoning trajectory
3:  t ← 0

4:  // Core Agent Loop
5:  while not TERMINATED do
6:      // Select action from policy π_θ
7:      a_t ~ π_θ(a | s_t)           // LLM-parameterized policy
8:      
9:      if a_t = GRAPH_SEARCH then
10:         context ← LOCAL_SEARCH(G_u, q)    // GraphRAG retrieval
11:         H_t+1 ← H_t ∪ {context}
12:     
13:     else if a_t = COMMUNITY_LOOKUP then
14:         communities ← GET_RELEVANT_COMMUNITIES(G_u, q)
15:         H_t+1 ← H_t ∪ {communities}
16:     end if
17:     
18:     // Epistemic Verification (Self-Reflection)
19:     v_t ~ P_φ(v | q, H_t) ∈ {SUFFICIENT, INSUFFICIENT}
20:     
21:     if v_t = INSUFFICIENT then
22:         // Trigger Fallback Agent for Visual Refinement
23:         E_q ← GET_RELEVANT_ENTITIES(G_u, q)  // Visual entities
24:         a_vis ← ONLINE_VISUAL_REFINEMENT(q, E_q)
25:         H_t+1 ← H_t ∪ {a_vis}    // Ephemeral context injection
26:     end if
27:     
28:     s_t+1 ← (q, H_t+1)
29:     t ← t + 1
30: end while

31: r ← GENERATE_RESPONSE(q, H_t)
32: return r
─────────────────────────────────────────────────────────────────
```

**Implementation**: [`backend/services/conversation.py`](backend/services/conversation.py)

#### Algorithm 3: Online Visual Refinement (t_VIS)

```
Algorithm: ONLINE_VISUAL_REFINEMENT(q, E_q)
─────────────────────────────────────────────────────────────────
Input: Query q, Relevant entities E_q ⊂ G_u
Output: Visual evidence synthesis a_vis

1:  // Contextual Fetching
2:  raw_payloads ← ∅
3:  for each e ∈ E_q do
4:      if HAS_VISUAL_PAYLOAD(e) then
5:          c ← FETCH_RAW_CONTENT(e)    // Original image tensor
6:          raw_payloads ← raw_payloads ∪ {c}
7:      end if
8:  end for

9:  // Neural VQA Injection
10: answers ← ∅
11: for each c ∈ raw_payloads do
12:     a_c ← M_vis(q, c)               // Multimodal LLM inference
13:     answers ← answers ∪ {a_c}
14: end for

15: // Aggregate visual evidence
16: a_vis ← AGGREGATE(answers)          // Natural language synthesis
17: return a_vis
─────────────────────────────────────────────────────────────────
```

**Implementation**: [`backend/services/analyzer.py`](backend/services/analyzer.py)

---

## Repository Structure

```
.
├── backend/                      # FastAPI application
│   ├── app.py                   # Main entry point, lifespan management
│   ├── core/                    # Core utilities
│   │   ├── config.py           # Centralized configuration
│   │   ├── deps.py             # Dependency injection
│   │   ├── jwt_auth.py         # JWT authentication
│   │   └── security.py         # Security utilities
│   ├── models/
│   │   └── payloads.py         # Pydantic request/response models
│   ├── routers/
│   │   ├── auth.py             # OAuth endpoints
│   │   ├── routers.py          # Main API routes
│   │   └── sync_runtime.py     # Sync management
│   └── services/
│       ├── pkg_population.py   # Φ_M and Φ_C implementation
│       ├── initialize_graphRag.py  # Community detection & GraphRAG
│       ├── graphRag_chat.py    # GraphRAG local search
│       ├── conversation.py     # Core Agent (Δ_Core) & Fallback Agent
│       ├── analyzer.py         # Visual Refinement (t_VIS)
│       ├── img_description.py  # Captioning operator (τ)
│       ├── img_location.py     # Geolocation extraction
│       └── google_*.py         # Google service integrations
│
├── libs/                         # External libraries (vendored)
│   └── llm_graph_builder/       # Knowledge graph construction library
│       ├── functions.py         # Public API (upload, extract, postprocess)
│       ├── src/                 # Core implementation
│       │   ├── main.py          # Entry point for extraction pipeline
│       │   ├── create_chunks.py # Text chunking (f_chunk)
│       │   ├── chunkid_entities.py  # Entity extraction (f_KGC)
│       │   ├── make_relationships.py # Relationship inference
│       │   ├── communities.py   # Leiden community detection
│       │   ├── post_processing.py # Embeddings & vector indexes
│       │   ├── graphDB_dataAccess.py # Neo4j graph operations
│       │   └── shared/          # Utilities & constants
│       ├── requirements.txt     # Library-specific dependencies
│       └── LICENSE              # Apache 2.0 License
│       
├── config/                       # Configuration files
│   └── hyperparameters.yaml     # Experiment hyperparameters
│
├── data/                         # Runtime data directory
│   ├── tmp/                     # Temporary processing files
│   └── sync/                    # Sync state persistence
│   
├── scripts/                      # Utility scripts
│   ├── setup_reproducibility.sh # Environment setup & validation
│   ├── download_dataset.py      # Dataset downloader
│   ├── generate_jwt.py          # JWT token generator
│   └── get_google_code.py       # OAuth flow helper
│
├── .github/workflows/           # CI/CD pipelines
│   └── reproducibility.yml      # Reproducibility validation
│
├── images/                       # Architecture diagrams
├── benchmark/                    # PersonalQA-71-100 dataset
├── pyproject.toml               # Dependencies (uv/pip)
├── uv.lock                      # Locked dependency versions
├── docker-compose.yml           # Service orchestration (Neo4j Enterprise)
├── docker-compose.community.yml # Alternative config (Neo4j Community)
├── Dockerfile                   # Multi-stage build
├── example.env                  # Environment template
└── LICENSE                      # MIT License
```

### LLM Graph Builder Integration

EpisTwin integrates a modified version of [LLM Graph Builder](https://github.com/neo4j-labs/llm-graph-builder) (Apache 2.0 License) for knowledge graph construction. The library is vendored directly under `libs/llm_graph_builder/` (not a submodule) to ensure reproducibility and ease of deployment.

#### Key Functions Used

| Function | Purpose | Used In |
|----------|---------|---------||
| `upload_file()` | Creates source nodes in Neo4j | `pkg_population.py` |
| `extract_knowledge_graph_from_file()` | LLM-based triple extraction (f_KGC) | `pkg_population.py` |
| `post_processing()` | Entity embeddings, vector indexes, communities | `pkg_population.py` |
| `delete_document_and_entities()` | Graph cleanup for re-sync | `pkg_population.py` |
| `execute_cypher_query()` | Direct Cypher query execution | `pkg_population.py`, `analyzer.py` |

#### Code Example: PKG Population Pipeline

```python
# backend/services/pkg_population.py
from libs.llm_graph_builder.functions import (
    upload_file,
    extract_knowledge_graph_from_file,
    post_processing,
    execute_cypher_query
)

async def llm_kg_builder_extract(fileName, database, content, ...):
    return await extract_knowledge_graph_from_file(
        file_name=fileName,
        content=content,
        uri=URI,
        userName=USERNAME,
        password=PASSWORD,
        database=database,
        model_name=EXTRACTION_MODEL_NAME,
        model_env_value=EXTRACTION_LLM_CONFIG,
        token_chunk_size=int(os.getenv("TOKENS_PER_CHUNK", 300)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 20)),
        # ... additional parameters
    )
```

#### Modifications from Upstream

The vendored library includes the following modifications:
- Multi-tenant database support via dynamic `database` parameter
- Custom model configuration format (`provider,model,base_url,api_key`)
- Integration with EpisTwin's environment variable system
- Additional error handling for production deployments

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | Required for type hints and async features |
| Docker | ≥ 24.0 | With Docker Compose v2 |
| [uv](https://github.com/astral-sh/uv) | Latest | Fast Python package manager |
| CUDA | ≥ 11.8 | Optional, for local model inference |

### Step 1: Install uv Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone and Install Dependencies

```bash
git clone https://github.com/sisinflab/EPANSA-orchestrator.git
cd EPANSA-orchestrator

# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### Step 3: Configure Google OAuth

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **OAuth 2.0 Client IDs**
5. Choose **Web application** as the application type
6. Set authorized redirect URIs to `http://localhost:5000/auth/callback`
7. Download the `client_secret.json` file
8. Place `client_secret.json` in the project root directory

### Step 4: Environment Setup

```bash
# Copy environment template
cp example.env .env

# Edit .env with your configuration (see Configuration section)
```

### Step 5: Launch Services

```bash
# Start all services (PostgreSQL, Neo4j, API)
docker compose up -d

# Verify services are running
docker compose ps

# View API logs
docker compose logs -f api
```

**Service Endpoints:**
- API: `http://localhost:5000`
- API Documentation: `http://localhost:5000/docs`
- Neo4j Browser: `http://localhost:7474`

---

## Configuration

### Environment Variables

#### Database Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_USER` | PostgreSQL username | `epansa` | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | - | Yes |
| `POSTGRES_DB` | Database name for token storage | `epansa_tokens` | Yes |
| `DATABASE_URL` | Full PostgreSQL connection URL | Auto-generated | No |
| `NEO4J_URI` | Neo4j Bolt connection URI | `bolt://neo4j:7687` | Yes |
| `NEO4J_USER` | Neo4j username | `neo4j` | Yes |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes |

#### Security

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENCRYPTION_KEY` | Fernet key for token encryption | - | Yes |
| `JWT_SECRET` | HS256 signing secret (dev only) | - | Yes |

#### Google OAuth

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_CLIENT_SECRETS_FILE` | Path to OAuth credentials | `client_secret.json` | Yes |
| `GOOGLE_CLIENT_ID` | OAuth client ID | From JSON file | No |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret | From JSON file | No |
| `GOOGLE_REDIRECT_URI` | OAuth callback URL | `http://localhost/` | Yes |

### Model Configurations

EpisTwin uses a unified configuration format for all LLM/VLM components:

```
provider,model_name,base_url,api_key
```

#### Supported Providers

| Provider | Config Prefix | Notes |
|----------|---------------|-------|
| Groq | `groq` | Fast inference, tool calling support |
| Google Gemini | `gemini` | Supports thinking budget |
| Ollama | `ollama` | Local models |
| OpenAI | `openai` | GPT-4, GPT-4V |
| Anthropic | `anthropic` | Claude models |

#### Model Role Configuration

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `EXTRACTION_LLM_CONFIG` | Entity extraction (Φ_C) | `groq,llama-3.3-70b-versatile,...` |
| `COMMUNITIES_LLM_CONFIG` | Community summarization (f_SUM) | `gemini,gemini-2.0-flash,...` |
| `CHATBOT_LLM_CONFIG` | GraphRAG response generation | `gemini,gemini-2.0-flash,...` |
| `AGENT_LLM_CONFIG` | Core Agent policy (π_θ) | `groq,qwen/qwen3-32b,...` |
| `IMG_ANALYSIS_VL_CONFIG` | Visual refinement (M_vis) | `groq,meta-llama/llama-4-scout-17b-16e-instruct,...` |
| `CHATBOT_EMBEDDING_CONFIG` | Community embeddings | `gemini,text-embedding-004,...` |

### Hyperparameters

#### PKG Population (Φ_C)

| Parameter | Description | Default | Range Explored |
|-----------|-------------|---------|----------------|
| `TOKENS_PER_CHUNK` | Token chunk size for text processing | 300 | [100, 500] |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks | 20 | [10, 50] |
| `NUMBER_OF_CHUNKS_TO_COMBINE` | Chunks combined for entity extraction | 3 | [1, 5] |
| `MAX_TOKEN_CHUNK_SIZE` | Max tokens per extraction call | 4000 | [2000, 8000] |
| `WORDS_FOR_BIG_FILE` | Threshold for large file handling | 500000 | - |

#### Community Detection (Leiden Algorithm)

| Parameter | Description | Default |
|-----------|-------------|---------|
| Resolution | Modularity resolution parameter | 1.0 |
| Min Community Size | Minimum nodes per community | 5 |

#### Reasoning Engine

| Parameter | Description | Default |
|-----------|-------------|---------|
| Max Iterations | Maximum reasoning steps | 10 |
| Temperature (π_θ) | Agent policy temperature | 0.7 |
| Top-K (Local Search) | Retrieved communities | 5 |

---

## Reproducing Experiments

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB |
| GPU | - | NVIDIA GPU ≥16GB VRAM (for local models) |
| Storage | 50 GB | 100 GB SSD |

> **Note**: EpisTwin primarily uses cloud-based LLM APIs. GPU is only required if running local models via Ollama.

### Step 1: Setup Environment

```bash
# Ensure all services are running
docker compose up -d

# Verify Neo4j is ready
docker compose exec neo4j cypher-shell -u neo4j -p <password> "RETURN 1"
```

### Step 2: Populate PKG with Test Data

```bash
# Authenticate with Google (opens browser)
python scripts/get_google_code.py

# Trigger full sync for a user
curl -X POST http://localhost:5000/sync/trigger \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json"
```

### Step 3: Run Benchmark Evaluation

```bash
# Run PersonalQA-71-100 evaluation
python -m benchmark.evaluate \
  --dataset benchmark/personalqa_71_100.json \
  --output results/evaluation_results.json

# Generate metrics report
python -m benchmark.metrics --input results/evaluation_results.json
```

### Step 4: Analyze Results

Results are evaluated using multiple judge models as described in the paper. Output includes:
- Accuracy per question category
- Reasoning trace analysis
- Visual refinement trigger frequency

---

## Dataset and Evaluation

The evaluation benchmark and knowledge base are located in the `evaluation/` directory. See [evaluation/README.md](evaluation/README.md) for detailed documentation following OpenScience standards.

### Knowledge Base

The Personal Knowledge Graph source data resides in `evaluation/knowledge_base/`:

| Directory | Content Description |
|-----------|---------------------|
| `epistwin_docs/` | PDF documents (tickets, receipts, manuals) |
| `epistwin_images/` | Visual data (photos from trips, documents) |
| `epistwin_jsontxt/` | Structured JSON metadata for graph nodes |

**Supported Data Types:**

| Source App | Example File | Content Description |
|:-----------|:-------------|:--------------------|
| Calendar | `event_8.txt` | Single/recurring events (meetings, gym) |
| Communication | `phoneCall_3.txt` | Call logs with duration and timestamps |
| Contacts | `contact_LucasSmith.txt` | Address book entries |
| Media (Meta) | `photo_20250615.txt` | Image EXIF data (location, time) |
| Documents | `doc_1.txt` | Metadata linking to files in `epistwin_docs` |
| Notes | `note_1_content.txt` | Unstructured text (diaries, to-do lists) |

### Benchmark Dataset

The `IJCAI_Test_Dataset.csv` contains ground truth for evaluation with questions requiring cross-modal reasoning:

| Column | Description |
|--------|-------------|
| `QUESTION` | Natural language query |
| `TARGET ANSWER` | Ground truth based on knowledge base |
| `EpisTwin ANSWER` | Model's generated response (for evaluation) |

**Sample Question Types:**
- *Temporal*: "Did I wake up before the alarm last Sunday?"
- *Cross-Modal*: "Where was I on June 1st at 20:00?" (requires photo metadata)
- *Reasoning*: "Why was I feeling stressed last Friday?" (correlates calendar with notes)

### Evaluation Pipeline

```bash
# 1. Run LLM-as-a-Judge evaluation (Prometheus rubric)
cd evaluation
python evalu.py

# 2. Normalize and merge results from multiple judges
python merge_normalization.py

# 3. Calculate Inter-Annotator Agreement metrics
python all_metrics_normalized.py
```

**Metrics Computed:**
- Cohen's Kappa (Quadratic)
- Gwet's AC1 (robust to high-agreement paradox)
- Spearman's Correlation
- Percentage Agreement

---

## API Reference

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | Initiate Google OAuth flow |
| `/auth/callback` | GET | OAuth callback handler |
| `/auth/me` | GET | Get current user info |

### Synchronization

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sync/trigger` | POST | Trigger manual PKG sync |
| `/sync/status` | GET | Get sync status |

### Reasoning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Query the reasoning engine |
| `/chat/history` | GET | Get conversation history |

### Full API documentation available at `/docs` when the server is running.

---

## Troubleshooting

### Docker Issues

**Problem**: Neo4j fails to start  
**Solution**: Ensure enterprise license acceptance:
```bash
docker compose down -v
# Verify NEO4J_ACCEPT_LICENSE_AGREEMENT=yes in docker-compose.yml
docker compose up -d
```

**Problem**: API cannot connect to databases  
**Solution**: Wait for databases to be ready:
```bash
docker compose logs neo4j | grep "Started"
docker compose logs postgres | grep "ready to accept connections"
```

### Model Configuration

**Problem**: LLM calls failing  
**Solution**: Verify API keys in `.env`:
```bash
# Test Groq connection
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

### PKG Population

**Problem**: Entity extraction produces empty results  
**Solution**: Check `MAX_TOKEN_CHUNK_SIZE` is sufficient for your content:
```bash
# Increase token limit
MAX_TOKEN_CHUNK_SIZE=8000
```

---

## Reproducibility

This section documents the resources and procedures required to reproduce the experimental results presented in the paper, following [IJCAI 2026 Reproducibility Guidelines](https://2026.ijcai.org/reproducibility/).

### Quick Start (Reproducibility Setup)

```bash
# 1. Clone the repository
git clone https://github.com/sisinflab/EPANSA-orchestrator.git
cd EPANSA-orchestrator

# 2. Run reproducibility setup script
chmod +x scripts/setup_reproducibility.sh
./scripts/setup_reproducibility.sh

# 3. Configure environment
cp example.env .env
# Edit .env with your API keys (see Configuration section)

# 4. Launch services
docker compose up -d
```

### System Requirements

| Component | Minimum | Recommended | Paper Experiments |
|-----------|---------|-------------|-------------------|
| CPU | 8 cores | 16+ cores | AMD EPYC 7763 (64 cores) |
| RAM | 16 GB | 32 GB | 128 GB |
| GPU | - | NVIDIA ≥16GB VRAM | - (API-based inference) |
| Storage | 50 GB SSD | 100 GB SSD | 500 GB NVMe |
| OS | Linux/macOS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### Software Dependencies

| Dependency | Version | Purpose |
|------------|---------|--------|
| Python | ≥ 3.11 | Runtime |
| Docker | ≥ 24.0 | Container orchestration |
| Docker Compose | v2 | Service management |
| uv | Latest | Package management |
| Neo4j | 5.x | Graph database |
| PostgreSQL | 15 | Token storage |

### Neo4j Configuration Options

EpisTwin supports both Neo4j Enterprise and Community Edition:

**Enterprise Edition** (default, recommended for production):
```bash
docker compose up -d
```

**Community Edition** (no license required, suitable for reproduction):
```bash
docker compose -f docker-compose.community.yml up -d
```

> **Note**: Community Edition does not include APOC and GDS plugins. Some advanced graph algorithms may have reduced performance, but core functionality remains intact.

### Hyperparameter Configuration

All hyperparameters used in paper experiments are documented in `config/hyperparameters.yaml`:

```yaml
# PKG Population (Φ_C)
pkg_population:
  tokens_per_chunk: 300        # Range explored: [100, 500]
  chunk_overlap: 20            # Range explored: [10, 50]
  chunks_to_combine: 3         # Range explored: [1, 5]
  max_token_chunk_size: 4000   # Range explored: [2000, 8000]

# Community Detection (Leiden)
community_detection:
  resolution: 1.0
  min_community_size: 5

# Reasoning Engine
reasoning:
  max_iterations: 10
  temperature: 0.7
  top_k_communities: 5
```

### API Keys Required

| Provider | Purpose | Environment Variable |
|----------|---------|---------------------|
| Groq | Entity extraction, Agent policy | `GROQ_API_KEY` |
| Google Gemini | Community summarization, Response generation | `GEMINI_API_KEY` |
| Google Cloud | OAuth for data access | `client_secret.json` |

### Reproducibility Checklist

- [ ] **Code Availability**: All source code is provided in this repository
- [ ] **Dependencies**: Locked versions in `uv.lock` and `requirements.txt`
- [ ] **Configuration**: Complete environment template in `example.env`
- [ ] **Hyperparameters**: Documented in `config/hyperparameters.yaml`
- [ ] **Hardware**: Computing infrastructure documented above
- [ ] **Data**: PersonalQA-71-100 benchmark (available in `evaluation/` directory)
- [ ] **External Libraries**: LLM Graph Builder vendored directly (Apache 2.0 license)

### Validation

Run the CI reproducibility check locally:

```bash
# Validate environment setup
./scripts/setup_reproducibility.sh --check-only

# Run integration tests
python -m pytest tests/ -v
```

---

## Citation

If you use EpisTwin in your research, please cite:

```bibtex
@inproceedings{epistwin2026,
  title     = {EpisTwin: Neuro-Symbolic Personal Knowledge Graphs for Trustworthy Personal AI},
  author    = {[Authors]},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-26)},
  year      = {2026},
  note      = {To appear}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [GraphRAG](https://github.com/microsoft/graphrag) for community detection and summarization
- [Neo4j](https://neo4j.com/) for graph database infrastructure

---

<p align="center">
  <i>EpisTwin: Grounding Personal AI in Verifiable Knowledge</i>
</p>
