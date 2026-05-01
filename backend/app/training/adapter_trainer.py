from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

from backend.app.llm.config_loader import load_llm_config
from backend.app.training.config import PlannerTrainerConfig


class AdapterTrainingImportError(RuntimeError):
    pass


class TimeBudgetStopCallback:
    def __init__(self, max_training_seconds: Optional[int]) -> None:
        self.max_training_seconds = max_training_seconds
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_training_seconds is None:
            return control
        if time.time() - self.start_time >= self.max_training_seconds:
            control.should_training_stop = True
            control.should_save = True
        return control


@dataclass
class AdapterTrainingPlan:
    role: str
    dataset_path: str
    base_model_ref: str
    base_model_path: str
    adapter_output_path: str
    adapter_name: str


class AdapterTrainer:
    def __init__(self, trainer_config: PlannerTrainerConfig) -> None:
        self.trainer_config = trainer_config
        self.output_dir = Path(trainer_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_plan(self, role: str) -> AdapterTrainingPlan:
        dataset_path = self.output_dir / f"{role}_training_examples.jsonl"
        adapter_config = self.trainer_config.adapter_for_role(role)
        local_provider = load_llm_config().providers["local"]
        base_model_path = local_provider.resolve_local_base_path(self.trainer_config.base_model_name)
        adapter_output_path = Path(self.trainer_config.finetuned_model_root) / self.trainer_config.base_model_name / role
        adapter_output_path.mkdir(parents=True, exist_ok=True)
        return AdapterTrainingPlan(
            role=role,
            dataset_path=str(dataset_path),
            base_model_ref=self.trainer_config.base_model_name,
            base_model_path=base_model_path,
            adapter_output_path=str(adapter_output_path),
            adapter_name=adapter_config.adapter_name,
        )

    def train_role(self, role: str) -> dict[str, Any]:
        plan = self.build_plan(role)
        dataset_path = Path(plan.dataset_path)
        if not dataset_path.exists() or dataset_path.stat().st_size == 0:
            return {"role": role, "trained": False, "reason": "no_training_examples"}

        try:
            from datasets import load_dataset
            from peft import LoraConfig
            from peft import PeftModel
            from peft import TaskType
            from peft import get_peft_model
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer
            from transformers import DataCollatorForLanguageModeling
            from transformers import Trainer
            from transformers import TrainerCallback
            from transformers import TrainingArguments
        except ImportError as exc:
            raise AdapterTrainingImportError(
                "Adapter training requires transformers, datasets, and peft. "
                "Use environment.finetune.yml."
            ) from exc

        class _StopCallback(TrainerCallback):
            def __init__(self, max_training_seconds: Optional[int]) -> None:
                self.helper = TimeBudgetStopCallback(max_training_seconds)

            def on_step_end(self, args, state, control, **kwargs):
                return self.helper.on_step_end(args, state, control, **kwargs)

        tokenizer = AutoTokenizer.from_pretrained(plan.base_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            plan.base_model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        adapter_output = Path(plan.adapter_output_path)
        resume_checkpoint = None
        if self.trainer_config.resume_from_checkpoint and (adapter_output / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter_output), is_trainable=True, adapter_name=plan.role)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.trainer_config.lora_r,
                lora_alpha=self.trainer_config.lora_alpha,
                lora_dropout=self.trainer_config.lora_dropout,
                target_modules=self.trainer_config.target_modules,
            )
            model = get_peft_model(model, peft_config)

        dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        def _format_messages(example: dict[str, Any]) -> dict[str, Any]:
            messages = example["messages"]
            rendered = []
            for message in messages:
                rendered.append(f"{message['role'].upper()}: {message['content']}")
            text = "\n\n".join(rendered)
            return {"text": text}

        dataset = dataset.map(_format_messages)

        def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
            encoded = tokenizer(
                example["text"],
                truncation=True,
                max_length=self.trainer_config.max_seq_length,
                padding="max_length",
            )
            encoded["labels"] = list(encoded["input_ids"])
            return encoded

        dataset = dataset.map(_tokenize, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=str(adapter_output),
            overwrite_output_dir=False,
            num_train_epochs=self.trainer_config.num_train_epochs,
            per_device_train_batch_size=self.trainer_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.trainer_config.gradient_accumulation_steps,
            learning_rate=self.trainer_config.learning_rate,
            warmup_steps=self.trainer_config.warmup_steps,
            logging_steps=self.trainer_config.logging_steps,
            save_steps=self.trainer_config.save_steps,
            save_total_limit=2,
            report_to=[],
            remove_unused_columns=False,
            fp16=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[_StopCallback(self._max_training_seconds())],
        )

        if self.trainer_config.resume_from_checkpoint:
            checkpoints = sorted(adapter_output.glob("checkpoint-*"))
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        model.save_pretrained(str(adapter_output))
        tokenizer.save_pretrained(str(adapter_output))
        metadata = {
            "role": role,
            "adapter_name": plan.adapter_name,
            "base_model_ref": plan.base_model_ref,
            "base_model_path": plan.base_model_path,
            "dataset_path": plan.dataset_path,
        }
        (adapter_output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
        return {
            "role": role,
            "trained": True,
            "base_model_ref": plan.base_model_ref,
            "base_model_path": plan.base_model_path,
            "adapter_output_path": plan.adapter_output_path,
            "dataset_path": plan.dataset_path,
            "resume_checkpoint": resume_checkpoint,
        }

    def _max_training_seconds(self) -> Optional[int]:
        if self.trainer_config.max_training_minutes is None:
            return None
        return int(self.trainer_config.max_training_minutes * 60)
