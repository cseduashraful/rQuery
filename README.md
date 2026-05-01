# DuckDB Predictive Agent

This repository contains a modular MVP implementation scaffold for a safe predictive reasoning
agent over DuckDB databases.

## Components

- `backend/`: FastAPI application, metadata store, agent pipeline, tests
- `frontend/`: Streamlit demo UI
- `config/`: model and runtime configuration
- `config/training/`: fine-tuning loop configuration
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

## Fine-Tuning Loop

The repository now includes a solver-critic fine-tuning scaffold focused on improving the task
planner from benchmark trajectories.

Main pieces:

- `backend/app/training/`: benchmark loader, execution engine, evaluator, trajectory store, and planner dataset builder
- `backend/app/agent/critic.py`: prompt-sufficiency critic
- `backend/app/agent/prompt_builder.py`: modular prompt integration for initial and critic-requested evidence
- `config/training/relbench_finetune_config.example.json`: example dataset/task configuration
- `scripts/run_finetune_loop.py`: loop runner

Run it with:

```bash
python scripts/run_finetune_loop.py --config config/training/relbench_finetune_config.example.json
```

The loop:

1. randomly picks a configured dataset
2. randomly picks a task from that dataset
3. randomly picks an entity from the task train split
4. builds a leakage-safe planner context
5. gathers pre-cutoff evidence
6. evaluates the final answer against ground truth
7. stores trajectories and planner-training examples in `output/training/`

## Notes

- The implementation is intentionally modular so the LLM provider, query sandbox, and profiling
  strategy can evolve independently.
- The local LLM path is configuration-driven and can be filled in later.
- Full environment notes, including PyTorch and local-LLM guidance, are in
  [docs/ENVIRONMENT.md](/Users/mdashrafulislam/paper_drafts/rQuery/docs/ENVIRONMENT.md).
