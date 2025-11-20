
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from pathlib import Path
import torch

# --------------------
# Einstellungen
# ---------------------

DATA_PATH = Path("data/processed/dataset.jsonl")
OUTPUT_DIR = Path("outputs/final_model_cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LEN = 512


print(f'Lade Dataset: {DATA_PATH.resolve()}')
dataset = load_dataset("json", data_files=str(DATA_PATH))["train"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# fallback für Sonderfälle
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(ex):
    # SFT_Format: Instruction + Input + "Antwort:" => Target = output
    prompt = f"{ex['instruction']}\n\n{ex['input']}\nAntwort:"
    return {
        "input_ids": tokenizer(
            prompt,
            max_length=MAX_LEN,
            padding="max_length",
        )["input_ids"],
        "attention_mask": tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )["attention_mask"],
        "labels": tokenizer(
            ex["output"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )["input_ids"],
    }
print("Tokenisierte dataset...")
tokenized_dataset = dataset.map(format_example,remove_columns=dataset.column_names)


# CPU-Modell
device = "cpu"
touch_dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=False,
)

# Lora minimal halten
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.5,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Trainingsparameter

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.0,
    learning_rate=5e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    bf16=False,
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Training...")
trainer.train()

print("Saving model...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(OUTPUT_DIR)


print(f"Fertig! Adapter gespeichert unter: {OUTPUT_DIR.resolve()}")