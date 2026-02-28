from mpmath.libmp.libintmath import ifac2
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import logging

from src.config import ModelConfig, LoRAConfig


class ModelSetup:
    """Kümmert sich um Modell- und Tokenizer-Setup"""

    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig, logger: logging.Logger):
        self.model_config = model_config
        self.lora_config = lora_config
        self.logger = logger

    def load_tokenizer(self):
        """Lädt und konfiguriert Tokenizer"""
        self.logger.info(f"⚙️ Lade Tokenizer: {self.model_config.base_model}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code
        )

        # PAD Token Setup für TinyLlama
        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                self.logger.info(f"✓ pad_token = unk_token ({tokenizer.unk_token})")
            else:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.info(f"✓ pad_token = eos_token ({tokenizer.eos_token})")

        # Padding-Seite für Llama
        tokenizer.padding_side = self.model_config.padding_side
        self.logger.info(f"✓ Padding-Seite: {tokenizer.padding_side}")

        # Token-Info
        self.logger.info(f"✓ Vocab Size: {len(tokenizer)}")
        self.logger.info(f"✓ BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        self.logger.info(f"✓ EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        self.logger.info(f"✓ PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

        return tokenizer

    def load_model(self):
        """Lädt Basis-Modell"""
        self.logger.info(f"⚙️ Lade Modell: {self.model_config.base_model}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype= torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None if self.model_config.device_map == "cpu" else self.model_config.device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=self.model_config.trust_remote_code
        )

        # Wichtig für Training
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        self.logger.info(f"✓ Modell geladen auf: {self.model_config.device_map}")
        self.logger.info(f"✓ Model dtype: {model.dtype}")

        model = prepare_model_for_kbit_training(model)
        return model

    def apply_lora(self, model):
        """Wendet LoRA auf Modell an"""
        self.logger.info("🔧 Aktiviere LoRA")

        lora_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias=self.lora_config.bias
        )

        # Model für Training vorbereiten
        model = get_peft_model(model, lora_config)

        # 🔑 WICHTIG: sicherstellen, dass LoRA-Parameter trainierbar sind
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # Parameter-Statistik
        from src.utils import print_trainable_params
        stats = print_trainable_params(model)
        self.logger.info(
            f"✓ Trainierbare Parameter: {stats['trainable']:,} / {stats['total']:,} "
            f"({stats['percentage']:.2f}%)"
        )
        self.logger.info(f"✓ LoRA Rank: {self.lora_config.r}")
        self.logger.info(f"✓ LoRA Alpha: {self.lora_config.alpha}")
        self.logger.info(f"✓ Target Modules: {self.lora_config.target_modules}")

        return model

    def setup(self):
        """Komplettes Setup: Tokenizer + Modell + LoRA"""
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        model = self.apply_lora(model)
        return model, tokenizer

