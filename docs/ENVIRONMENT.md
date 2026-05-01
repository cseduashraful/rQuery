# Environment Requirements

This project runs well in a Conda environment. The simplest path is to use the provided
`environment.yml`.

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

These are required for the current MVP:

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

`torch` is not required for the current OpenAI-backed MVP path, but it is included in the Conda
environment so the project is ready for future local-LLM work.

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

## Optional Local-LLM Extras

These are not required yet, but you will likely want some of them when you wire in a local model:

- `transformers`
- `accelerate`
- `sentencepiece`
- `safetensors`
- `protobuf`
- `llama-cpp-python`
- `vllm` for Linux/NVIDIA deployments

Recommended install examples:

```bash
pip install transformers accelerate sentencepiece safetensors
```

For `llama-cpp-python`:

```bash
pip install llama-cpp-python
```

Notes:

- `vllm` is usually a Linux/NVIDIA-oriented dependency and is not recommended as a default in the
  shared Conda file.
- `llama-cpp-python` is often the simplest local path for a single-machine setup.

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

When you later add a real local LLM backend such as a Llama 70B path, create one additional file
such as `environment.local-llm.yml` instead of overloading the baseline environment. That keeps the
OpenAI-backed MVP lightweight while still making heavy local inference setups reproducible.
