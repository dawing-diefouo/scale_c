# src/model_setup.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from src.config import ModelConfig, LoRAConfig


class ModelSetup:
    """
    Mistral-Style:
    - Tokenizer + Base-Model laden
    - LoRA auf Base-Model anwenden
    """

    def __init__(self, model_cfg: ModelConfig, lora_cfg: LoRAConfig):
        self.model_cfg = model_cfg
        self.lora_cfg = lora_cfg

    def load_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(
            self.model_cfg.base_model,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = self.model_cfg.padding_side
        return tok

    def load_base_model(self, torch_dtype=None, device_map=None):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.base_model,
            trust_remote_code=self.model_cfg.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        return model

    def apply_lora(self, model):
        lora = LoraConfig(
            r=self.lora_cfg.r,
            lora_alpha=self.lora_cfg.alpha,
            lora_dropout=self.lora_cfg.dropout,
            target_modules=self.lora_cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias=self.lora_cfg.bias,
        )
        model = get_peft_model(model, lora)

        # Logging-Info (ohne logger dependency)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✓ Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
        print(f"✓ target_modules: {self.lora_cfg.target_modules}")
        return model

    def setup(self, device: str, cuda_device: int, use_fp16: bool, use_bf16: bool):
        tok = self.load_tokenizer()

        if device == "cpu":
            dtype = torch.float32
            device_map = None
        else:
            torch.cuda.set_device(cuda_device)
            if use_bf16:
                dtype = torch.bfloat16
            elif use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float16  # default für GPU
            device_map = {"": f"cuda:{cuda_device}"}

        base = self.load_base_model(torch_dtype=dtype, device_map=device_map)
        model = self.apply_lora(base)

        if getattr(model, "gradient_checkpointing_enable", None) is not None:
            # enable wird im train.py anhand config gesetzt; hier nur verfügbar
            pass

        return model, tok
