<div align="center">
  <img src="docs/assets/hero.png" alt="Cerebro API Hero Banner" width="100%">

  <br>

  [![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)

  *An async, resilient, self-hosted autonomous research agent exposed via REST API.*
</div>

---

## 🧠 What is Cerebro API?

Cerebro is a next-generation backend service that provides a completely autonomous agent orchestration layer. Built entirely in Python using **FastAPI**, it allows you to dispatch complex research, mathematical, and reasoning tasks into the background and retrieve structured JSON reports when they finish.

**Why it's different:**
* **True Concurrency:** Tool executions (like heavy web searches or API calls) are dispatched concurrently using `asyncio.gather`, slashing execution time.
* **Strict Memory Isolation:** Each background job gets a globally unique `job_id`, structurally preventing cross-contamination of RAG (Retrieval-Augmented Generation) memory between different agent runs.
* **Bulletproof Resilience:** Core LLM routing and network tools are wrapped in `Tenacity` exponential backoff, ensuring momentary network blips never crash a long-running research task.
* **Granular Observability:** Integrated orchestration logging cleanly outputs per-iteration token consumption and exact dollar costs using `OpenAI` usage schemas.

---

## 🏛️ Architecture

```mermaid
graph TD
    User([Client / User]) -->|POST /research| API(FastAPI Endpoint)
    API -->|Returns 202 Job ID| User
    API -->|Spawns Background Task| Orchestrator[Agent Orchestrator]
    
    Orchestrator --> LLM{LLM (gpt-4o-mini)}
    
    LLM -->|Plan & Execute| Registry[Tool Registry]
    
    Registry -->|asyncio.gather| WebSearch[Web Search (Tenacity + Backoff)]
    Registry -->|asyncio.gather| Calc[Calculator]
    Registry -->|asyncio.gather| Memory[RAG Memory Engine]
    
    Memory --> Isolate[(Vector Store per job_id)]
    
    Orchestrator -->|Final Report| Result[(In-Memory Job Store)]
    User -->|GET /status/{job_id}| Result
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12+
- `uv` package manager (recommended)
- OpenAI API Key (or OpenRouter compatible key)

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CognitoServe/Cerebro-API.git
   cd Cerebro-API
   ```

2. Install dependencies via `uv`:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Setup environment variables:
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY
   ```

4. Run the server:
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

### Docker Deployment
```bash
docker build -t cerebro-api .
docker run -d -p 8000:8000 --env-file .env --name cerebro cerebro-api
```

---

## 📡 API Usage

### 1. Dispatch a Research Job
Send a POST request to dispatch the agent into the background.

```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"query": "Analyze the latest performance metrics of FastAPI vs Litestar."}'
```
**Response:**
```json
{
  "job_id": "5117f21d-597e-4eb6-9a13-3c5e76746472",
  "status": "pending",
  "message": "Research job submitted successfully."
}
```

### 2. Poll for Status
Check the status of your job using the `job_id`. The orchestrator prevents polling blocks and instantly returns the current state.

```bash
curl -X GET "http://localhost:8000/status/5117f21d-597e-4eb6-9a13-3c5e76746472"
```

**Response (When Finished):**
```json
{
  "job_id": "5117f21d-597e-4eb6-9a13-3c5e76746472",
  "status": "completed",
  "report": {
    "topic": "FastAPI vs Litestar Performance",
    "summary": "Litestar shows up to 20% faster throughput due to its tighter integration with MsgSpec...",
    "findings": [
      "Litestar relies on MsgSpec for serialization.",
      "FastAPI relies on Pydantic V2."
    ]
  }
}
```

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
