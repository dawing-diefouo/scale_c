from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Datenpfade und -einstellungen"""
    train_path: Path = Path("data/processed/train_data.jsonl")
    eval_path: Optional[Path] = None
    max_length: int = 1024  # H5P-JSONs sind länger


@dataclass
class ModelConfig:
    """Modell-spezifische Einstellungen"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device_map: str = "cpu"
    padding_side: str = "left"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA-Parameter"""
    r: int = 16  # Höher für komplexe JSON-Strukturen
    alpha: int = 32
    dropout: float = 0.1  # Niedriger als vorher
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Training-Parameter"""
    output_dir: Path = Path("outputs/final_model_cpu")
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 300
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    use_fp16: bool = False
    save_total_limit: int = 3
    max_grad_norm: float = 1.0


@dataclass
class Config:
    """Hauptkonfiguration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


