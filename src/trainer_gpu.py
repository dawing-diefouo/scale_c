from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from pathlib import Path
import json
import logging

from src.config import TrainingConfig


class ModelTrainer:
    """Kapselt die komplette Training-Logik"""

    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def create_training_args(self, has_eval: bool):
        """Erstellt TrainingArguments (vereinfachte Version, ohne Evaluation-Strategie)"""
        return TrainingArguments(
            output_dir=str(self.config.output_dir),

            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            max_grad_norm=self.config.max_grad_norm,

            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_steps=self.config.warmup_steps,

            logging_steps=self.config.logging_steps,
            logging_dir=str(self.config.output_dir / "logs"),
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,

            # 🔥 GPU-Settings
            fp16=self.config.use_fp16,
            bf16=False,
            no_cuda=False,  # ← MUSS False sein

            gradient_checkpointing=False,  # empfohlen für GPU-Speicher

            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=2,
        )

    def train(self, model, tokenizer, train_dataset, eval_dataset=None):
        """Führt Training durch"""
        self.logger.info("🚀 Starte Training")

        # Setup
        training_args = self.create_training_args(has_eval=eval_dataset is not None)

        # Data Collator (wichtig: setzt PAD in Labels auf -100)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, nicht Masked LM
        )

        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=tokenizer,
        )

        # Training starten
        try:
            trainer.train()
            self.logger.info("✓ Training erfolgreich abgeschlossen")
        except Exception as e:
            self.logger.error(f"❌ Training-Fehler: {e}", exc_info=True)
            raise

        # Speichern
        self.logger.info("💾 Speichere Modell")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.config.output_dir))
        tokenizer.save_pretrained(str(self.config.output_dir))

        # Stats
        self._save_training_stats(trainer)

        self.logger.info(f"🎉 Training fertig! → {self.config.output_dir}")
        return trainer

    def _save_training_stats(self, trainer):
        """Speichert Training-Statistiken"""
        stats_path = self.config.output_dir / "training_stats.json"
        with open(stats_path, "w", encoding='utf-8') as f:
            json.dump(trainer.state.log_history, f, indent=2)
        self.logger.info(f"✓ Stats gespeichert: {stats_path}")

