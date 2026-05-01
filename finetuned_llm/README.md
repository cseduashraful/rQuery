# Fine-Tuned Local Models

This directory is reserved for locally stored fine-tuned model copies or role-adapter-ready model
directories by parameter size.

Expected layout:

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

Runtime resolution rule for the local provider:

1. If `finetuned_llm/<size>/<role>` exists and is non-empty, use it as the adapter for that role.
2. Otherwise fall back to the original base path configured in `config/llm_config.json`.
3. If `finetuned_llm/<size>/config.json` exists, it is treated as a merged fine-tuned model copy.

The directory itself is tracked in git, but model contents are ignored.
