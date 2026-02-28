from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Datenpfade und -einstellungen"""
    train_path: Path = PROJECT_ROOT / "data/processed/train_data.jsonl"
    eval_path: Optional[Path] = None
    max_length: int = 1024  # ✅ FIX: 512 reicht für H5P Multiple-Choice


@dataclass
class ModelConfig:
    """Modell-spezifische Einstellungen"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device_map: str = "auto"
    padding_side: str = "left"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA-Parameter"""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1

    # ✅ OPTIMIERT: Nur wichtigste Module für kleines Modell
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj"  # Reicht für die meisten Fälle
    ])
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Training-Parameter"""
    output_dir: Path = PROJECT_ROOT / "outputs/final_model_gpu"
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4  # ✅ FIX: Höher für LoRA
    warmup_steps: int = 100  # ✅ FIX: Reduziert für kleines Dataset
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    use_fp16: bool = True
    save_total_limit: int = 2  # ✅ FIX: Weniger Checkpoints
    max_grad_norm: float = 1.0


@dataclass
class Config:
    """Hauptkonfiguration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)