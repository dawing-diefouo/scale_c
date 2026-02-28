# src/trainer.py
import json
from pathlib import Path

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.config import TrainingConfig


class ModelTrainer:
    """Mistral-Style Trainer-Wrapper"""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    def build_args(self, has_eval: bool) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(self.cfg.output_dir),

            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,

            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=self.cfg.weight_decay,
            max_grad_norm=self.cfg.max_grad_norm,

            logging_steps=self.cfg.logging_steps,

            save_strategy="steps",
            save_steps=self.cfg.save_steps,
            save_total_limit=self.cfg.save_total_limit,

            evaluation_strategy=self.cfg.evaluation_strategy if has_eval else "no",
            eval_steps=self.cfg.eval_steps if has_eval else None,

            fp16=self.cfg.use_fp16,
            bf16=self.cfg.use_bf16,

            gradient_checkpointing=self.cfg.gradient_checkpointing,
            remove_unused_columns=False,

            report_to="none",
        )

    def train(self, model, tokenizer, train_dataset, eval_dataset=None):
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        args = self.build_args(has_eval=eval_dataset is not None)

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Speichern (Adapter + tokenizer + config)
        trainer.save_model(str(self.cfg.output_dir))
        tokenizer.save_pretrained(str(self.cfg.output_dir))

        # Logs speichern
        stats_path = Path(self.cfg.output_dir) / "training_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

        print(f"🎉 Training fertig: {self.cfg.output_dir}")
        return trainer
