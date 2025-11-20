from datasets import load_dataset, Dataset
import logging
from pathlib import Path
from typing import Optional

from src.config import DataConfig


class DatasetLoader:
    """Lädt und validiert Datasets"""

    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_train_data(self) -> Dataset:
        """Lädt Trainingsdaten"""
        self.logger.info(f"📥 Lade Training: {self.config.train_path}")

        if not self.config.train_path.exists():
            raise FileNotFoundError(f"Nicht gefunden: {self.config.train_path}")

        dataset = load_dataset("json", data_files=str(self.config.train_path))["train"]
        self._validate_dataset(dataset)
        self.logger.info(f"✓ {len(dataset)} Beispiele geladen")

        # Zeige erstes Beispiel
        self.logger.info(f"📝 Beispiel-Instruction: {dataset[0]['instruction'][:100]}...")
        self.logger.info(f"📝 Beispiel-Output: {dataset[0]['output'][:100]}...")

        return dataset

    def load_eval_data(self) -> Optional[Dataset]:
        """Lädt optionale Evaluationsdaten"""

        # Wenn eval_path nicht gesetzt ist → keine Eval-Daten
        if self.config.eval_path is None:
            self.logger.info("ℹ️ Keine Evaluationsdaten konfiguriert (eval_path=None)")
            return None

        # Wenn eval_path existieren müsste aber nicht existiert → Info und weiter
        if not self.config.eval_path.exists():
            self.logger.info(f"ℹ️ Eval-Daten nicht gefunden: {self.config.eval_path}")
            return None

        # Wenn eval_path da ist → laden
        self.logger.info(f"📥 Lade Evaluation: {self.config.eval_path}")
        dataset = load_dataset("json", data_files=str(self.config.eval_path))["train"]
        self._validate_dataset(dataset)
        self.logger.info(f"✓ {len(dataset)} Eval-Beispiele geladen")

        return dataset

    def _validate_dataset(self, dataset: Dataset):
        """Validiert Datenstruktur"""
        required_keys = {'instruction', 'output'}
        sample_keys = set(dataset[0].keys())

        if not required_keys.issubset(sample_keys):
            raise ValueError(
                f"Dataset benötigt Keys: {required_keys}, "
                f"gefunden: {sample_keys}"
            )

        # Prüfe ob output valides JSON ist
        from src.utils import is_valid_json
        if not is_valid_json(dataset[0]['output']):
            self.logger.warning("⚠️ Erstes Beispiel hat invalides JSON im output-Feld")

