import json
from datasets import Dataset
from transformers import PreTrainedTokenizer


class DataPreprocessor:
    """Verantwortlich für Formatierung und Tokenisierung"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_h5p_example(self, instruction: str, output):
        """Baut das Chatformat"""

        # Falls output ein JSON-String ist → in dict umwandeln
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except:
                pass

        # JSON sauber serialisieren
        output_json = json.dumps(output, ensure_ascii=False)

        system_message = (
            "Antworte ausschließlich mit GENAU EINEM JSON-Objekt. "
            "KEINE Erklärungen, KEIN zusätzlicher Text."
        )

        prompt = (
            f"<|system|>\n{system_message}</s>\n"
            f"<|user|>\n{instruction}</s>\n"
            f"<|assistant|>\n{output_json}</s>"
        )

        return prompt

    def tokenize_function(self, examples):
        texts = [
            self.format_h5p_example(inst, out)
            for inst, out in zip(examples["instruction"], examples["output"])
        ]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Labels: Kopie der Eingaben
        labels = encodings["input_ids"].clone()

        # Maskiere ALLES vor dem Assistant-Output
        for i, text in enumerate(texts):
            assistant_token_start = text.index("<|assistant|>")
            # finde Länge bis zum Assistant-Token
            prefix_ids = self.tokenizer(
                text[:assistant_token_start],
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"][0]

            labels[i, : prefix_ids.shape[0]] = -100

        encodings["labels"] = labels
        return {k: v.tolist() for k, v in encodings.items()}

    def process_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenisierung",
        )
