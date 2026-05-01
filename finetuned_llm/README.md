# Fine-Tuned Local Models

This directory is reserved for locally stored fine-tuned model copies or role-adapter-ready model
directories by parameter size.

Expected layout:

```text
finetuned_llm/
  1b/
  3b/
  8b/
  70b/
```

Runtime resolution rule for the local provider:

1. If `finetuned_llm/<size>` exists and is non-empty, use it first.
2. Otherwise fall back to the original base path configured in `config/llm_config.json`.

The directory itself is tracked in git, but model contents are ignored.

