
import logging
from pathlib import Path
import json

from src.config import Config


def setup_logging(output_dir: Path, name: str = __name__) -> logging.Logger:
    """Konfiguriert Logging"""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


def print_trainable_params(model):
    """Zeigt Anzahl trainierbarer Parameter"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    return {
        "trainable": trainable,
        "total": total,
        "percentage": percentage
    }


def save_config(config: Config, output_dir: Path):
    """Speichert Konfiguration als JSON"""
    from dataclasses import asdict

    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, default=str)


def is_valid_json(text: str) -> bool:
    """Prüft ob Text valides JSON ist"""
    try:
        json.loads(text)
        return True
    except:
        return False