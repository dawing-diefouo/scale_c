<<<<<<< HEAD
# 🧠 H5P-Generator mit TinyLlama + LoRA (CPU-freundlich)

Dieses Projekt finetuned ein LLM (TinyLlama-1.1B-Chat) darauf, **valide H5P Multiple-Choice content.json Dateien** automatisch zu generieren.
Es beinhaltet:

* Extraktion von content.json aus bestehenden .h5p Dateien
* Erzeugung von Instruction-Pairs
* Training mit LoRA auf CPU
* Validierte JSON-Ausgabe
* Erstellung fertiger .h5p Pakete

---

# 📂 Projektstruktur

```
scale_c/
│
├── data/
│   ├── raw/                # Originale .h5p Dateien
│   ├── processed/
│   │      ├── train_data.jsonl  # Trainingsdatensätze
│   └── h5p/                # Hier entstehen generierte .h5p Dateien
│
├── outputs/
│   └── final_model_cpu/    # Fine-Tuned Modell + Tokenizer
│
├── src/
│   ├── extract_h5p.py      # Extrahiert content.json aus raw .h5p
│   ├── preprocessing.py    # erstellt Chat-Prompt + tokenisiert
│   ├── model_setup.py      # lädt Modell + konfiguriert LoRA
│   ├── trainer.py          # Training-Loop
│   ├── data_loader.py      # lädt Dataset
│   ├── h5p_validator.py    # prüft JSON-Struktur
│   ├── utils.py
│   └── train.py            # Haupt-Trainingsskript
│
└── test_model_cpu.py       # Inference + H5P-Generierung
```

---

# 📥 1. Datenerstellung aus bestehenden H5P Dateien

Alle H5P-Quellen in:

```
data/raw/*.h5p
```

Dann ausführen:

```bash
python src/extract_h5p.py
```

Ergebnis:

```
data/processed/train_data.jsonl
```

Jede Zeile enthält:

```json
{
  "instruction": "...",
  "output": "{json-string}"
}
```

---

# 🛠 2. Finetuning (TinyLlama + LoRA)

Training starten:

```bash
python -m src.train
```

Das Skript:

* lädt train_data.jsonl
* erstellt Chat-Prompts:

```
<|system|>Du bist ein H5P-Generator...</s>
<|user|>Instruction</s>
<|assistant|>OutputJSON</s>
```

* tokenisiert
* führt LoRA-Training durch
* speichert alles nach:

```
outputs/final_model_cpu/
```

---

# ⚙️ 3. Hyperparameter (einfach erklärt)

### **num_epochs = 3**

Wie oft das Modell alle Trainingsdaten sieht.
Mehr Daten → weniger Epochen notwendig.

### **learning_rate = 1e-4**

Wie stark das Modell bei jedem Schritt lernt.
Niedriger = stabiler, besser für JSON-Aufgaben.

### **warmup_steps = 200**

Modell beginnt mit kleiner Lernrate → schützt vor instabilem Training.

### **batch_size = 1**

Notwendig auf CPU.

### **gradient_accumulation_steps = 4**

Simuliert effektiv Batch-Size 4 → stabilisiert das Training.

### **max_length = 1024**

H5P content.json sind lang → 1024 Tokens optimal.

---

# 🔧 LoRA-Parameter

### **r = 16**

Lernkapazität der Adapter.

### **alpha = 32**

Skalierung der LoRA-Updates.

### **dropout = 0.1**

Verhindert Overfitting bei kleinen Datensätzen.

### **target_modules**

LoRA wird in folgenden Llama-Modulen aktiv:

```
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

---

# 🧪 4. Inference – Erzeuge ein valides H5P content.json

Mit:

```bash
python test_model_cpu.py
```

Der Prompt nutzt exakt dasselbe Format wie im Training:

```
<|system|>Du bist ein H5P-Generator...</s>
<|user|>Frage...</s>
<|assistant|>
```

Das Modell erzeugt:

* valides JSON
* H5P-Struktur mit question, answers, behaviour, overallFeedback
* validiert durch `H5PValidator`
* speichert fertiges H5P-Paket in:

```
data/h5p/generated_mc.h5p
```

---

# 📦 5. Erzeugte H5P Datei öffnen

Du kannst die Datei direkt testen auf:

👉 [https://h5p.org/multichoice](https://h5p.org/multichoice)

Einfach **Upload** wählen.

---

# 🚀 6. Empfehlung für gute Ergebnisse

Damit TinyLlama verlässlich gültige H5P-JSON generiert:

* **mindestens 200 Trainingsdatensätze** verwenden
* **Epochen auf 3 reduzieren** (sonst Overfitting)
* **do_sample=False** und **temperature=0.0** beim Inference setzen
* **Validator verwenden**, um Fehler sofort zu erkennen

Mit 200 Beispielen wird die Qualität **dramatisch** besser.

---

# 🤝 Weiterentwicklung

Empfohlene Erweiterungen:

* Automatischer H5P-Kursgenerator
* Weitere H5P-Typen (Drag&Drop, Fill in the Blanks)
* Auto-Augmentation für mehr Trainingsdaten
* Web-Interface (Gradio/Streamlit)

---

MIT License – frei erweiterbar.
=======
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
>>>>>>> 4883651dd8b773f588b5de801d971fe629bcefe2
