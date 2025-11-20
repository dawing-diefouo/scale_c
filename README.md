# 🧠 CyberSecurity Instruction-Finetuning  
**Fine-Tuning eines Sprachmodells mit H5P-Lerninhalten (Instruction-FT + PEFT/LoRA)**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/LoRA-Adapter-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧩 Projektübersicht
Dieses Projekt untersucht, wie **H5P-Lerninhalte** (z. B. Quizfragen) genutzt werden können,  
um ein **Sprachmodell** mit **Instruction-Finetuning** und **Parameter-Efficient Fine-Tuning (PEFT/LoRA)** zu verbessern.  
Das Beispielthema ist *Cybersicherheit*.

Ziel: Ein Modell, das Lernfragen beantworten und erklären kann – auf Basis realer H5P-Daten.

---

## 🗂️ Projektstruktur

```bash
project/
│
├── data/
│   ├── raw/                # Original H5P-Dateien
│   ├── processed/          # Extrahierte JSONs
│   └── dataset.jsonl       # Finale Trainingsdaten für Instruction-FT
│
├── src/
│   ├── extract_h5p.py      # Skript: H5P → JSONL
│   ├── train_instruction_ft.py  # Training mit HuggingFace + PEFT
│   └── utils/              # Hilfsfunktionen
│
├── notebooks/
│   ├── data_preview.ipynb  # Dateninspektion
│   └── training_eval.ipynb # Evaluation des Trainings
│
├── configs/
│   ├── training_config.yaml
│   └── model_config.json
│
├── outputs/
│   ├── checkpoints/        # Modellgewichte
│   ├── logs/               # TensorBoard / W&B Logs
│   └── eval_results.json
│
├── README.md
├── requirements.txt
└── .gitignore
