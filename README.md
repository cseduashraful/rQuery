# DuckDB Predictive Agent

This repository contains a modular MVP implementation scaffold for a safe predictive reasoning
agent over DuckDB databases.

## Components

- `backend/`: FastAPI application, metadata store, agent pipeline, tests
- `frontend/`: Streamlit demo UI
- `config/`: model and runtime configuration
- `docs/`: implementation and extension documentation in LaTeX
- `sample_data/`: placeholder area for example DuckDB files

## Quick Start

1. Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate duckdb-predictive-agent
```

2. Set `OPENAI_API_KEY` if using the OpenAI provider.
3. Run the API:

```bash
uvicorn backend.app.main:app --reload
```

4. Run the demo UI:

```bash
streamlit run frontend/streamlit_app.py
```

## Notes

- The implementation is intentionally modular so the LLM provider, query sandbox, and profiling
  strategy can evolve independently.
- The local LLM path is configuration-driven and can be filled in later.
- Full environment notes, including PyTorch and local-LLM guidance, are in
  [docs/ENVIRONMENT.md](/Users/mdashrafulislam/paper_drafts/rQuery/docs/ENVIRONMENT.md).
