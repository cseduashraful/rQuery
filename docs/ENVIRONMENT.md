# Environment Requirements

This project runs well in Conda environments. The lightest path is to use `environment.yml`, but
there are now separate files for local inference and real fine-tuning too.

## Environment Files

- `environment.yml`
  Lightweight app/runtime/scaffold environment
- `environment.local-inference.yml`
  For mostly local LLM inference with a stored base model plus adapters
- `environment.finetune.yml`
  For actual adapter fine-tuning jobs

## Recommended Baseline

- OS: macOS, Linux, or Windows with Conda or Mamba
- Python: `3.11`
- SQLite: bundled via Conda
- DuckDB: Python package
- FastAPI stack: `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy`
- Demo UI: `streamlit`
- LLM provider: `openai`
- Testing: `pytest`

## Conda Environment

Create and activate:

```bash
conda env create -f environment.yml
conda activate duckdb-predictive-agent
```

## Core Python Packages

These are required for the current lightweight app/runtime scaffold:

- `alembic`
- `duckdb`
- `fastapi`
- `httpx`
- `openai`
- `pydantic`
- `pydantic-settings`
- `python-multipart`
- `sqlalchemy`
- `sqlglot`
- `streamlit`
- `uvicorn`
- `requests`
- `pytest`

## PyTorch

`torch` is not required for the pure OpenAI-backed path, but it is included in the Conda
environments so the project is ready for local-LLM inference and future fine-tuning work.

Current `environment.yml` choice:

- `pytorch`
- `cpuonly`

This is the safest default because it works broadly and avoids CUDA-specific install failures.

## If You Want GPU PyTorch Later

### NVIDIA CUDA

Replace the CPU-only PyTorch entries in `environment.yml` with the appropriate CUDA variant from
the PyTorch channel, for example:

```yaml
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - pytorch
  - pytorch-cuda=12.1
```

### Apple Silicon

PyTorch on Apple Silicon usually works through the standard Conda or pip install, using MPS at
runtime where supported. In that case you can remove `cpuonly` and install `pytorch` alone.

## Local Inference And Fine-Tuning Packages

If you plan to run mostly local LLM inference, `transformers`-style packages are usually needed.

Those are included in:

- `environment.local-inference.yml`
- `environment.finetune.yml`

The fine-tuning environment also includes:

- `datasets`
- `peft`
- `trl`

Use the dedicated files instead of overloading the light baseline environment.

## System Tools

Helpful tools to have available:

- `git`
- `make` or `just` if you later add task runners
- a LaTeX distribution if you want to compile the implementation book in `docs/`

For LaTeX compilation:

- macOS: MacTeX
- Linux: TeX Live
- Windows: MiKTeX or TeX Live

## Environment Variables

For the OpenAI-backed path:

```bash
export OPENAI_API_KEY=your_key_here
```

Optional future variables for local backends may include:

- `LOCAL_LLM_HOST`
- `LOCAL_LLM_PORT`
- `LOCAL_LLM_MODEL_PATH`

The default config design also supports:

- one fine-tuned planner model
- one shared base model for the other inference roles

The training config also supports an adapter-oriented setup where one shared base model is reused
across roles and only role-specific adapters are updated during fine-tuning.

See also:

- [FINETUNE_ENVIRONMENT.md](/Users/mdashrafulislam/paper_drafts/rQuery/docs/FINETUNE_ENVIRONMENT.md)

## Build And Run

Run the API:

```bash
uvicorn backend.app.main:app --reload
```

Run the Streamlit UI:

```bash
streamlit run frontend/streamlit_app.py
```

Run tests:

```bash
python -m pytest
```

Generate sample data:

```bash
python scripts/create_sample_dbs.py
```

Run the planner fine-tuning loop:

```bash
python scripts/run_finetune_loop.py --config config/training/relbench_finetune_config.example.json
```

## Practical Recommendation

Use `environment.yml` as the default shared team environment.

Use the dedicated environment files instead of overloading the baseline environment. That keeps the
app/runtime scaffold lightweight while still making heavy local inference and cluster fine-tuning
setups reproducible.
