import json
from datasets import Dataset
from transformers import PreTrainedTokenizer


class DataPreprocessor:
    """Verantwortlich für Formatierung und Tokenisierung"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Sicherstellen, dass pad_token gesetzt ist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_h5p_example(self, tokenizer, instruction: str, output: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Antworte ausschließlich mit GENAU EINEM JSON-Objekt. KEIN Zusatztext."
            },
            {
                "role": "user",
                "content": instruction
            },
            {
                "role": "assistant",
                "content": output
            }
        ]

        # tokenize=False gibt den fertigen String zurück
        # add_generation_prompt=False, da wir den Assistant-Output (unser Label) mitgeben
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def tokenize_function(self, examples):
        """Tokenisiert Batch von Beispielen"""
        # 1. Formatiere alle Beispiele unter Verwendung des Chat-Templates
        # Wichtig: Übergeben Sie hier den Tokenizer an Ihre format_h5p_example
        texts = [
            self.format_h5p_example(self.tokenizer, inst, out)
            for inst, out in zip(examples['instruction'], examples['output'])
        ]

        # 2. Tokenisieren
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None  # Bleibt None für Datasets-Library
        )

        # 3. Labels hinzufügen (Kopie der input_ids)
        # Das ist für den SFTTrainer oder Trainer notwendig, damit er weiß, was er lernen soll
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]

        encodings["labels"] = labels
        return {k: v.tolist() for k, v in encodings.items()}

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Verarbeitet komplettes Dataset"""
        processed = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenisierung",
            load_from_cache_file=False
        )

        # Format für PyTorch setzen
        processed.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

        return processed