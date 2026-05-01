# Fine-Tune Environment

This document describes the heavier environment used for local-model inference and adapter
fine-tuning.

## Why Separate Files

The repository now has three environment layers:

- `environment.yml`
  App runtime and scaffold only.
- `environment.local-inference.yml`
  For running mostly local LLM inference with a stored base model plus adapters.
- `environment.finetune.yml`
  For real adapter fine-tuning jobs.

This split exists because local inference and fine-tuning need `transformers`-style packages,
while the lighter app scaffold does not.

## Local Inference Environment

Use this when:

- the app will mostly use local LLMs
- you want to load a stored base model
- you want to attach role-specific adapters at inference time
- you do not necessarily want to run gradient updates

Create it with:

```bash
conda env create -f environment.local-inference.yml
conda activate duckdb-predictive-agent-local-inference
```

It includes:

- `transformers`
- `accelerate`
- `sentencepiece`
- `safetensors`
- `protobuf`
- `torch`

That is enough for many local-inference setups built around:

- one shared base model
- one adapter per role

## Fine-Tuning Environment

Use this when:

- you want to actually fine-tune a local stored model
- you want adapter-based updates
- you want to stop after some wall-clock time and resume later

Create it with:

```bash
conda env create -f environment.finetune.yml
conda activate duckdb-predictive-agent-finetune
```

It adds the training stack:

- `transformers`
- `accelerate`
- `datasets`
- `peft`
- `trl`
- `tensorboard`

## Recommended Fine-Tuning Mechanism

For this project, the preferred mechanism is:

- one shared base model
- one adapter per role
- train only one role adapter at a time
- keep the base model and all other adapters frozen

That means the thing that gets updated after a training run is usually:

- adapter weights

not the entire base model.

This is the right fit for your workflow because you may want to:

- fine-tune for one hour
- save the updated adapter weights
- run inference with those new adapter weights
- come back later and continue training

## Checkpoint And Resume Model

The intended workflow is:

1. Load a stored base model.
2. Load one role adapter, or initialize one if needed.
3. Train that adapter for a time budget, such as one hour.
4. Save the adapter checkpoint.
5. Register the new adapter version in your training metadata.
6. Use that adapter for later inference.
7. Resume training from that adapter checkpoint in the future.

This is different from retraining from scratch each time.

## What “Updated Weights” Means Here

In the adapter-based design, the updated weights are usually:

- LoRA or PEFT adapter weights

The base model weights normally remain unchanged.

That is still a real model update in practice, because inference uses:

- base model
- plus the latest adapter weights

If you later want to merge adapter weights into the base model for deployment, that can be done as
a separate deployment step.

## Suggested Directory Layout For Real Training

A practical cluster-side layout could be:

```text
models/
  base/
    llama-70b/
  adapters/
    task_planner/
      run_001/
      run_002/
    sql_explorer/
      run_001/
    final_predictor/
      run_001/
    critic/
      run_001/
```

This makes it easy to:

- keep the base model immutable
- version adapters by role
- resume fine-tuning from the last checkpoint
- evaluate one adapter without disturbing the others

Inside this repository, the runtime also reserves:

```text
finetuned_llm/
  1b/
    task_planner/
    sql_explorer/
    final_predictor/
    critic/
  3b/
    task_planner/
    sql_explorer/
    final_predictor/
    critic/
  8b/
    task_planner/
    sql_explorer/
    final_predictor/
    critic/
  70b/
```

The local provider resolves a parameter size alias by using the configured base path first and then
checking for a role adapter in `finetuned_llm/<size>/<role>`. If a role adapter is present, it is
attached to the already loaded base model for that role. If `finetuned_llm/<size>/config.json`
exists, that directory can also be treated as a merged fine-tuned model copy.

If multiple runtime roles resolve to the same local path, the intended runtime behavior is to load
that model once and reuse the same loaded instance across steps.

## Time-Budgeted Fine-Tuning

Your desired workflow of “fine-tune for one hour” is very reasonable.

The training job should support:

- max wall-clock duration
- periodic checkpoint saves
- resume from latest checkpoint

That means one cluster run might:

- fine-tune the `task_planner` adapter for one hour
- save `task_planner/run_003`
- stop cleanly

Then a later run can:

- load `task_planner/run_003`
- continue for another hour
- save `task_planner/run_004`

## What The Current Repo Already Supports

The current repository already supports:

- one shared base model concept
- one adapter per role in the training config
- role-specific training example export
- random role sampling in the fine-tune loop

What it does not yet do is run the actual adapter training itself. That cluster-side step is still
meant to be connected through your future trainer hook.

Update:

The repository now includes a real adapter trainer implementation based on
`transformers + peft`. The main entrypoint is:

```bash
python scripts/run_adapter_training.py --config config/training/relbench_finetune_config.example.json --role task_planner
```

The built-in trainer is still intended primarily for a cluster or heavy fine-tuning environment,
not the lightweight app/runtime environment.

## Practical Recommendation

Use:

- `environment.local-inference.yml` for local-model inference and development
- `environment.finetune.yml` for cluster-side adapter training

This keeps the basic app environment small while still supporting your intended local-LLM and
incremental fine-tuning workflow.
