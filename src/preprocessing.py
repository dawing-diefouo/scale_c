from datasets import Dataset
from transformers import PreTrainedTokenizer


class DataPreprocessor:
    """Verantwortlich für Formatierung und Tokenisierung"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_h5p_example(self, instruction: str, output: str) -> str:


        system_message = (
            "Du bist ein H5P-Content-Generator. Deine Aufgabe ist es, "
            "interaktive Lernmaterialien im H5P-JSON-Format zu erstellen. "
            "Antworte NUR mit validem JSON, ohne zusätzliche Erklärungen oder Text."
        )

        prompt = (
            f"<|system|>\n{system_message}</s>\n"
            f"<|user|>\n{instruction}</s>\n"
            f"<|assistant|>\n{output}</s>"
        )

        return prompt

    def tokenize_function(self, examples):
        """Tokenisiert Batch von Beispielen"""
        # Formatiere alle Beispiele
        texts = [
            self.format_h5p_example(inst, out)
            for inst, out in zip(examples['instruction'], examples['output'])
        ]

        # Tokenisieren
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

        # Labels = Input IDs (für Causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Verarbeitet komplettes Dataset"""
        return dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenisierung"
        )
