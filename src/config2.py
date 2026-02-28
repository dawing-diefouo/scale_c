# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    train_path: Path = Path("data/processed/dataset.jsonl")
    eval_path: Path = Path("data/processed/eval_dataset.jsonl")
    max_length: int = 1024


@dataclass
class ModelConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    trust_remote_code: bool = False
    padding_side: str = "left"  # für CausalLM oft sinnvoll


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"

    # MANUELL ändern für MLP / ATTN / MIXED / FULL
    target_modules: List[str] = field(default_factory=lambda: [
        # Mixed (Attention + MLP) als Default
        "q_proj", "k_proj", "v_proj", "o_proj"
        #"gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    output_dir: Path = Path("outputs/run")

    num_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2

    learning_rate: float = 2e-4
    warmup_steps: int = 200
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    evaluation_strategy: str = "steps"
    eval_steps: int = 4

    use_fp16: bool = False
    use_bf16: bool = False
    gradient_checkpointing: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
