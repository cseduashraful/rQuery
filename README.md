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

If you plan to run mostly local LLM inference, use:

```bash
conda env create -f environment.local-inference.yml
conda activate duckdb-predictive-agent-local-inference
```

If you plan to run actual adapter fine-tuning jobs, use:

```bash
conda env create -f environment.finetune.yml
conda activate duckdb-predictive-agent-finetune
```

## Fine-Tuning Loop

The repository now includes a solver-critic fine-tuning scaffold focused on improving the task
planner from benchmark trajectories.

Main pieces:

- `backend/app/training/`: benchmark loader, execution engine, evaluator, trajectory store, and planner dataset builder
- `backend/app/training/adapter_trainer.py`: actual PEFT/LoRA adapter trainer
- `backend/app/agent/critic.py`: prompt-sufficiency critic
- `backend/app/agent/prompt_builder.py`: modular prompt integration for initial and critic-requested evidence
- `config/training/relbench_finetune_config.example.json`: example dataset/task configuration
- `scripts/run_finetune_loop.py`: loop runner
- `scripts/run_adapter_training.py`: actual adapter training entrypoint

Run it with:

```bash
python scripts/run_finetune_loop.py --config config/training/relbench_finetune_config.example.json
```

The loop:

1. randomly picks a configured dataset
2. randomly picks a task from that dataset
3. randomly picks an entity from the task train split
4. randomly picks a trainable role such as `task_planner`, `sql_explorer`, `final_predictor`, or `critic`
5. builds a leakage-safe planner context
6. gathers pre-cutoff evidence
7. evaluates the final answer against ground truth
8. stores trajectories and role-specific adapter training examples in `output/training/`

To run actual adapter training after examples have been generated:

```bash
python scripts/run_adapter_training.py --config config/training/relbench_finetune_config.example.json --role task_planner
```

Set `trainer.mode` to `peft_lora` if you want the fine-tune loop to trigger adapter training after
trajectory generation.

## Model Configuration

The config layer supports a practical deployment pattern where:

- `task_planner_model` points to a fine-tuned planner model
- `shared_base_model` is reused for `sql_explorer`, `final_predictor`, `summary`, and optionally `critic`

That lets you specialize the planner without paying the cost of loading unrelated models for every
other role.

The fine-tuning loop also supports adapter-oriented training with:

- one shared base model name
- one adapter per role
- random role sampling per episode
- role-specific exported JSONL files for later cluster training

For local models, the repo now supports parameter-size aliases such as `1b`, `3b`, and `8b`.
When the provider is `local`, inference resolves models like this:

1. use the configured base path for the selected size alias
2. check `finetuned_llm/<size>/<role>` for a role adapter
3. if that adapter folder exists and is non-empty, attach it to the shared loaded base model
4. if `finetuned_llm/<size>/config.json` exists, it can also act as a merged fine-tuned model copy
5. otherwise fall back to the original base path only

If multiple roles resolve to the same local model path, the runtime now loads that local model once
and reuses the same in-memory instance across planner, explorer, predictor, and critic calls.

## Running With Your Preferred LLM Config

By default, the app reads:

```bash
config/llm_config.json
```

If that is the config you want, just edit it and run:

```bash
uvicorn backend.app.main:app --reload
```

If you want to use a different config file, set `LLM_CONFIG_PATH`:

```bash
export LLM_CONFIG_PATH=/absolute/path/to/my_llm_config.json
uvicorn backend.app.main:app --reload
```

You can also put this in `.env`:

```env
LLM_CONFIG_PATH=/absolute/path/to/my_llm_config.json
```

To enable actual LLM calls, also set:

```bash
export ENABLE_LLM_CALLS=true
```

So a full example is:

```bash
export LLM_CONFIG_PATH=/absolute/path/to/my_llm_config.json
export ENABLE_LLM_CALLS=true
uvicorn backend.app.main:app --reload
```

Current caveats:

- the main `/predict` path still uses heuristic fallback in parts of the current scaffold
- local provider path resolution is implemented, but full local-model execution is not yet wired end to end

## Notes

- The implementation is intentionally modular so the LLM provider, query sandbox, and profiling
  strategy can evolve independently.
- The local LLM path is configuration-driven and can be filled in later.
- Full environment notes, including PyTorch and local-LLM guidance, are in
  [docs/ENVIRONMENT.md](/Users/mdashrafulislam/paper_drafts/rQuery/docs/ENVIRONMENT.md).
- The heavier local-inference and adapter fine-tuning environment story is in
  [docs/FINETUNE_ENVIRONMENT.md](/Users/mdashrafulislam/paper_drafts/rQuery/docs/FINETUNE_ENVIRONMENT.md).
