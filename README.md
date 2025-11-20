# H5P-Generator: Fachliche und Wissenschaftliche Dokumentation

## 1. Einleitung
Dieses Projekt untersucht die Anwendung von **Instruction-Fine-Tuning** und **Parameter-Efficient Fine-Tuning (PEFT)**, insbesondere LoRA, zur Erzeugung **valider H5P-Multiple-Choice-Inhalte** mittels eines kompakten Sprachmodells (TinyLlama-1.1B-Chat). Ziel ist die automatische Generierung vollständig strukturierter *content.json*-Dateien, wie sie im H5P-Ökosystem zur Darstellung interaktiver Lerninhalte verwendet werden.

Der Fokus liegt auf einer präzisen und zuverlässigen Reproduktion der geforderten JSON-Struktur. Die Validierung erfolgt im sogenannten *Strict Mode*, in dem ausschließlich vollständig gültige Inhalte akzeptiert werden.

---

## 2. Projektstruktur
Die Implementierung ist modular aufgebaut, um Datenverarbeitung, Modelltraining und Validierung klar voneinander zu trennen.

```
scale_c/
│
├── data/
│   ├── raw/                # Originale .h5p Dateien
│   ├── processed/          # Extrahierte JSON-Daten
│   └── h5p/                # Generierte H5P-Dateien
│
├── outputs/
│   └── final_model_cpu/    # Trainiertes Modell
│
├── src/
│   ├── extract_h5p.py
│   ├── preprocessing.py
│   ├── model_setup.py
│   ├── trainer.py
│   ├── data_loader.py
│   ├── h5p_validator.py     # Strict Validator (keine Autokorrektur)
│   ├── utils.py
│   └── train.py
│
└── test_model_cpu.py        # Inferenzskript im Strict Mode
```

---

## 3. Datenextraktion und Aufbereitung
Zur Erstellung des Trainingskorpus werden vorhandene H5P-Dateien analysiert. Mithilfe von `extract_h5p.py` wird die Datei *content.json* extrahiert und in ein formatgerechtes JSONL-Format überführt.

### Ausführung:
```
python src/extract_h5p.py
```

Das resultierende Trainingsset befindet sich unter:
```
data/processed/train_data.jsonl
```

Jedes Element besteht aus:
- einer Instruction (Beschreibung der Aufgabe)
- dem zugehörigen H5P-Output (als String repräsentiertes JSON)

Diese Struktur ist kompatibel mit dem Supervised Fine-Tuning (SFT) von Chat-basierten Modellen.

---

## 4. Modelltraining
Das Fine-Tuning erfolgt auf CPU mithilfe des TinyLlama-Modells und LoRA-Adaptern. Die Trainingspipeline umfasst:

1. Laden und Validieren der Daten
2. Tokenisierung und Formatierung entsprechend des Chat-Templates:
```
<|system|> ... </s>
<|user|> Instruction </s>
<|assistant|> OutputJSON </s>
```
3. Training mit PEFT/LoRA
4. Speicherung des Modells und der Trainingsstatistiken

### Start des Trainings:
```
python -m src.train
```

Das trainierte Modell befindet sich in:
```
outputs/final_model_cpu/
```

---

## 5. Hyperparameter
Die folgenden Werte werden für das Instruction-Fine-Tuning empfohlen und sind empirisch geeignet für Datensätze ab ca. 200 Beispielen.

### 5.1 Trainingsparameter
- **Epochen (num_epochs):** 3  
  Reduzierung des Overfitting-Risikos.
- **Lernrate (learning_rate):** 1e-4  
  Stabilität beim Lernen strukturierter Ausgaben.
- **Warmup-Schritte (warmup_steps):** 200  
  Sanfter Anstieg der Lernrate.
- **Batch-Größe:** 1 (CPU-bedingt)
- **Gradient Accumulation:** 4  
  Effektive Batch-Größe = 4.
- **max_length:** 1024  
  Für umfangreiche content.json notwendig.

### 5.2 LoRA-Parameter
- **r = 16**  
  Adapterkapazität.
- **alpha = 32**  
  Skalierungsfaktor.
- **dropout = 0.1**  
  Reduziert Overfitting.
- **target_modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  
  Wichtige Komponenten des TinyLlama-Transformers.

---

## 6. Inferenzprozess (Strict Mode)
Der Inferenzprozess validiert streng, dass das Modell **selbstständig** eine korrekte *content.json* erzeugt.  
Eine automatische Reparatur findet **nicht** statt.

### Ausführung:
```
python test_model_cpu.py
```

Ergebnisse werden abgelegt unter:
```
data/h5p/generated_mc.h5p
```

Ein valides Ergebnis erfordert korrektes Auftreten folgender Felder:
- `question`
- `answers` (mindestens zwei Einträge)
- `correct` (mindestens ein Eintrag mit `true`)
- optional: `behaviour`, `overallFeedback`

Die Validierung erfolgt über `h5p_validator.py`, welches ausschließlich strukturelle Korrektheit überprüft.

---

## 7. Evaluierung
Ein valider Output ist notwendig, um den Fortschritt des Fine-Tunings zu messen.  
Es wird empfohlen:

- Datensatzgröße nach Bedarf zu erweitern
- mehrere Trainingsläufe durchzuführen
- die Trainingsstatistiken (`training_stats.json`) zu analysieren

---

## 8. Weiterentwicklung
Potenzielle Erweiterungen umfassen:

- Unterstützung zusätzlicher H5P-Typen (Drag & Drop, Fill-in-the-Blanks)
- automatisierte Generierung synthetischer Trainingsdaten
- Integration eines webbasierten Interfaces (z. B. Streamlit oder Gradio)
- quantitative Evaluation der JSON-Validitätsrate

---

## 9. Lizenz
Dieses Projekt steht unter der MIT-Lizenz und kann frei erweitert und modifiziert werden.
