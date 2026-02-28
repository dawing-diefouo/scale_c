# src/train.py
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from src.config2 import Config
from src.model_setup2 import ModelSetup
from src.preprocessing import DataPreprocessor
from src.trainer2 import ModelTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="outputs/attn_2")

    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--eval_path", type=str, default=None)

    # Quick overrides
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.training.output_dir = Path(args.output_dir)

    if args.train_path:
        cfg.data.train_path = Path(args.train_path)
    if args.eval_path:
        cfg.data.eval_path = Path(args.eval_path)

    if args.max_length is not None:
        cfg.data.max_length = args.max_length
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.training.per_device_train_batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.training.gradient_accumulation_steps = args.grad_accum

    cfg.training.use_fp16 = bool(args.fp16)
    cfg.training.use_bf16 = bool(args.bf16)
    cfg.training.gradient_checkpointing = not bool(args.no_grad_ckpt)

    print(f"🧠 base_model: {cfg.model.base_model}")
    print(f"📦 train_path: {cfg.data.train_path}")
    print(f"💾 output_dir: {cfg.training.output_dir}")
    print(f"🧩 target_modules: {cfg.lora.target_modules}")
    print(f"⚙️ max_length={cfg.data.max_length} epochs={cfg.training.num_epochs} lr={cfg.training.learning_rate}")
    print(f"⚙️ fp16={cfg.training.use_fp16} bf16={cfg.training.use_bf16} grad_ckpt={cfg.training.gradient_checkpointing}")

    # Load model/tokenizer
    setup = ModelSetup(cfg.model, cfg.lora)
    model, tokenizer = setup.setup(
        device=args.device,
        cuda_device=args.cuda_device,
        use_fp16=cfg.training.use_fp16,
        use_bf16=cfg.training.use_bf16,
    )
    # 1) Für PEFT + Gradient Checkpointing nötig:
    if cfg.training.gradient_checkpointing:
        # 1) Gradient checkpointing aktivieren
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

        # 2) WICHTIG: Embedding-Output muss requires_grad=True haben (sonst grad_fn Fehler)
        def _make_inputs_require_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    if cfg.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load dataset (jsonl)
    train_ds = load_dataset("json", data_files=str(cfg.data.train_path), split="train")
    eval_ds = None
    if cfg.data.eval_path and cfg.data.eval_path.exists():
        eval_ds = load_dataset("json", data_files=str(cfg.data.eval_path), split="train")
        cfg.training.evaluation_strategy = "steps"

    # Preprocess
    pre = DataPreprocessor(tokenizer, cfg.data.max_length)
    train_tok = pre.process_dataset(train_ds)
    eval_tok = pre.process_dataset(eval_ds) if eval_ds is not None else None

    # Train
    trainer = ModelTrainer(cfg.training)
    trainer.train(model, tokenizer, train_tok, eval_tok)

    # Save config snapshot
    snapshot = {
        "base_model": cfg.model.base_model,
        "target_modules": cfg.lora.target_modules,
        "max_length": cfg.data.max_length,
        "training": {
            "epochs": cfg.training.num_epochs,
            "batch_size": cfg.training.per_device_train_batch_size,
            "grad_accum": cfg.training.gradient_accumulation_steps,
            "lr": cfg.training.learning_rate,
            "fp16": cfg.training.use_fp16,
            "bf16": cfg.training.use_bf16,
            "grad_ckpt": cfg.training.gradient_checkpointing,
        },
    }
    with open(cfg.training.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print("✅ Done.")


if __name__ == "__main__":
    main()
