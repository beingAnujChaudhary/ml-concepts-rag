# ML Concepts Atlas

An end-to-end retrieval-augmented generation application that answers machine-learning questions from a curated Wikipedia knowledge base. It is designed for learners who want concise, source-backed explanations instead of searching through long articles. Every grounded answer also includes a Mermaid concept map to illustrate the relationships or process being explained.

## Architecture

The automated ingestion script fetches 13 machine-learning articles through the Wikipedia API, cleans and chunks the text, embeds chunks with `all-MiniLM-L6-v2`, and rebuilds an Elasticsearch index. At query time, GPT-4.1 rewrites the question, Elasticsearch performs hybrid BM25 and vector retrieval, and GPT-4.1 generates a context-only answer plus a Mermaid diagram. Streamlit provides the interface. PostgreSQL stores interactions, latency, and feedback; Grafana visualizes this monitoring data.

| Capability | Technology |
|---|---|
| Data source | Wikipedia API |
| LLM | OpenAI GPT-4.1 |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Knowledge base | Elasticsearch 8.14 hybrid search |
| Interface | Streamlit with Mermaid diagrams |
| Monitoring | PostgreSQL 15 and Grafana 11 |
| Runtime | Docker Compose |

## Prerequisites

Install Docker Desktop with Docker Compose and create an OpenAI API key with access to GPT-4.1. API usage incurs OpenAI charges. Never commit or paste your key into source code, screenshots, issues, or chat.

## Run locally

Clone the repository and enter its directory. Then create the local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `OPENAI_API_KEY` to a newly created key. The default model is `gpt-4.1`; it can be changed through `OPENAI_MODEL`.

Build and start the stack:

```bash
docker compose up -d --build
docker compose ps
```

Fetch Wikipedia content, create embeddings, and rebuild the Elasticsearch index:

```bash
docker compose exec app python data_ingestion/ingest.py
```

Open the services:

- Streamlit: http://localhost:8501
- Grafana: http://localhost:3000
- Elasticsearch: http://localhost:9200

Try questions such as “How does backpropagation work?”, “Compare supervised and unsupervised learning,” or “What is a random forest?” Expand **Show Sources** to inspect retrieved evidence and use the feedback buttons to populate monitoring data.

To inspect failures:

```bash
docker compose logs --tail=100 app
docker compose logs --tail=100 elasticsearch
docker compose logs --tail=100 postgres
```

To stop the project, use `docker compose down`. Add `-v` only when you intentionally want to delete indexed and monitoring data.

## Evaluation criteria

**Problem description (2):** The application provides focused, source-backed explanations and visual concept maps for people learning machine learning.

**Retrieval flow (2):** Questions are rewritten, embedded, retrieved with Elasticsearch hybrid search, and answered by GPT-4.1 from the top contexts.

**Retrieval evaluation (2):** `evaluation/evaluate_retrieval.ipynb` compares text, vector, and hybrid retrieval. Hybrid search is the production choice.

**LLM evaluation (2):** `evaluation/evaluate_llm.ipynb` compares multiple prompts with an LLM-as-a-judge workflow using OpenAI. Run it with a valid API key and record the results before submission.

**Interface (2):** Streamlit offers chat, expandable citations, feedback controls, and generated Mermaid concept diagrams.

**Ingestion pipeline (2):** `data_ingestion/ingest.py` automates fetching, cleaning, chunking, embedding, index creation, and indexing.

**Monitoring (2):** Feedback and latency are stored in PostgreSQL. The provisioned Grafana dashboard contains five panels: total queries, average response time, feedback distribution, queries over time, and recent queries.

**Containerization (2):** Streamlit, Elasticsearch, PostgreSQL, and Grafana run with Docker Compose. OpenAI is an external managed API.

**Reproducibility (2):** Dependencies and service versions are specified, the dataset is fetched from a public API, configuration is documented, and no secret is committed.

**Best practices:** Hybrid search and query rewriting are implemented. `rerank_results` is currently only an integration hook and must not be claimed as real document reranking until a reranker is implemented and evaluated.

## Evaluation and submission checklist

Before submission, execute both evaluation notebooks, save visible results, capture screenshots of the UI and five-panel Grafana dashboard, test the setup from a fresh clone, and verify that `.env` is not tracked. Submit the public GitHub repository URL and the exact final commit hash. Complete three peer reviews to receive the associated points.
