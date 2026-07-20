# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The documented execution path is Docker Compose; the application depends on Elasticsearch and PostgreSQL plus the external OpenAI API. Grafana is included for monitoring.

```bash
# Build and start the complete stack
docker compose up -d --build

# Inspect service status and logs
docker compose ps
docker compose logs -f app

# Fetch Wikipedia data, create embeddings, and rebuild the Elasticsearch index
docker compose exec app python data_ingestion/ingest.py

# Stop the stack (retain named-volume data)
docker compose down

# Stop the stack and remove Elasticsearch, PostgreSQL, and Grafana volumes
docker compose down -v
```

After startup, the Streamlit UI is at `http://localhost:8501`, Grafana at `http://localhost:3000`, and Elasticsearch at `http://localhost:9200`.

For a host-Python workflow with infrastructure running separately:

```bash
python -m pip install -r requirements.txt
python data_ingestion/ingest.py
streamlit run app/app.py
```

The Python modules accept `OPENAI_API_KEY`, `OPENAI_MODEL`, `ELASTIC_URL`, `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME`. Copy `.env.example` to `.env`; Docker Compose refuses to start the app when `OPENAI_API_KEY` is absent.

There is currently no configured automated test, lint, format, or build system beyond the Docker image build. Do not claim `pytest`, Ruff, Black, or similar commands are available unless their configuration and dependencies are added. The two notebooks under `evaluation/` are analysis artifacts rather than an automated test suite.

## Architecture

This is an end-to-end RAG application over a small, curated set of machine-learning Wikipedia pages. Docker Compose starts Streamlit, Elasticsearch, PostgreSQL, and Grafana on one network; GPT-4.1 is accessed through the external OpenAI API. Service hostnames and configuration are injected into the app container through `docker-compose.yaml`; when modules are run on the host, infrastructure defaults target localhost.

The offline ingestion path is `data_ingestion/ingest.py`: it fetches the configured `WIKI_TOPICS` through the `wikipedia` package, strips section markup, chunks article paragraphs, embeds each chunk with `all-MiniLM-L6-v2` (384 dimensions), and destructively recreates the Elasticsearch `ml_concepts` index before indexing the chunks. It also writes `sample_docs.json` relative to the process working directory. The `overlap` argument in `chunk_text` is currently unused.

The online request path starts in `app/app.py`. Streamlit keeps chat messages in session state and delegates each question to `app/rag.py`. GPT-4.1 rewrites the query, the same sentence-transformer used during ingestion embeds it, Elasticsearch performs combined BM25 and k-NN retrieval, and a reranking hook passes the top three contexts to GPT-4.1 for a context-only answer. A second grounded generation creates Mermaid flowchart code, which Streamlit renders in an isolated HTML component. The current `rerank_results` implementation is a no-op placeholder, not actual reranking.

`app/db.py` owns monitoring persistence. On Streamlit startup it creates the PostgreSQL `interactions` table if needed. Each response records the original query, rewritten query, answer, and latency; thumbs-up/down actions update the same row. Grafana provisioning under `grafana/provisioning/` points dashboards at this PostgreSQL data.

Retrieval and LLM experiments live in `evaluation/evaluate_retrieval.ipynb` and `evaluation/evaluate_llm.ipynb`. Keep evaluation logic aligned with production choices in `app/rag.py`, especially index name, embedding model, retrieval variants, prompt variants, and selected best approach.

## Coupling and operational constraints

Ingestion and retrieval must use the same Elasticsearch index (`ml_concepts`), embedding model, vector dimensionality, and document field names (`id`, `title`, `url`, `text`, `embedding`). A change on one side generally requires a corresponding change on the other and a fresh ingestion run.

`app/app.py` imports `db` and `rag` as sibling modules, so the supported entrypoint is `streamlit run app/app.py` from the repository root (as used by the Dockerfile). Importing it as a conventional package is not currently configured.

The app container can start before dependencies are ready because Compose `depends_on` controls startup order but no health checks are defined. Initial connection failures may therefore be transient. OpenAI generation requires network access, API credits, model access, and a valid `OPENAI_API_KEY` supplied through the untracked `.env` file.

Pinned Python dependencies are in `requirements.txt`, while external service versions are set in `docker-compose.yaml`. The model name defaults to `gpt-4.1` and can be overridden with `OPENAI_MODEL`.
