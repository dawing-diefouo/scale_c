# -*- coding: utf-8 -*-

import json
from typing import Optional
import re
import torch
from pathlib import Path

from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  #  KRITISCH: Für LoRA-Adapter!
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------
# Modellpfad
# --------------------------------------
MODEL_PATH = PROJECT_ROOT / "outputs" / "final_model_gpu"

print(f"Lade Modell aus: {MODEL_PATH}")

# ==================== WICHTIG: LoRA LADEN ====================

# 1. Lade BASIS-Modell
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f" Basis-Modell: {base_model_name}")
print(f" Lade LoRA-Adapter von: {MODEL_PATH}")

# 2. Lade Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Lade Basis-Modell
print(" Lade Basis-Modell...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=None,
    trust_remote_code=True
)

# 4.  KRITISCH: Lade LoRA-Adapter drauf
print(" Lade LoRA-Adapter...")
model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
model = model.merge_and_unload()   # 🔑 FEHLT BEI DIR
model.eval()
model.to(device)


print("Modell mit LoRA-Adapter geladen!")
print(f"   Device: {device}")

# Speicherordner für erzeugte H5P-Dateien
OUTPUT_DIR = Path("data/h5p")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# --------------------------------------
# Hilfsfunktionen
# --------------------------------------

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Antworte ausschließlich mit GENAU EINEM JSON-Objekt. KEINE Erklärungen, KEIN zusätzlicher Text."
        },
        {
            "role": "user",
            "content": f"Erstelle eine H5P-Multiple-Choice-Frage mit 4 Antwortmöglichkeiten und 1 richtigen Antwort(en), basierend auf folgender Frage: '{question}'."
        }
    ]

    # add_generation_prompt=True fügt den Header für den Assistant automatisch an
    # (z.B. <|assistant|>\n), damit das Modell weiß: Jetzt bin ich dran!
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def model_answer(question: str) -> str:
    """Ruft das Modell auf."""
    # Übergeben Sie den geladenen Tokenizer als erstes Argument
    prompt = build_prompt(tokenizer, question)

    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)


    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
        )

    full_response = tokenizer.decode(output[0], skip_special_tokens=False)

    # Extrahiere nur den Assistant-Teil
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1]
    else:
        response = full_response

    # Stoppe bei </s>
    if "</s>" in response:
        response = response.split("</s>")[0]

    return response.strip()




def extract_json(text):
    """Extrahiert das erste vollständige JSON-Objekt aus einem String."""
    try:
        # Suche nach dem ersten { und dem letzten }
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            return None

        json_str = text[start_idx:end_idx + 1]
        return json.loads(json_str)
    except Exception as e:
        print(f"Fehler beim Parsen: {e}")
        return None


def validate_h5p_structure(data: dict) -> Tuple[bool, str]:

    """Validiert H5P-Struktur"""
    # Pflichtfelder
    if 'question' not in data:
        return False, "Fehlendes Feld: question"

    if 'answers' not in data:
        return False, "Fehlendes Feld: answers"

    if not isinstance(data['question'], str):
        return False, "question muss ein String sein"

    if not isinstance(data['answers'], list):
        return False, "answers muss eine Liste sein"

    if len(data['answers']) != 4:
        return False, f"Erwarte 4 Antworten, gefunden: {len(data['answers'])}"

    # Prüfe Antworten
    correct_count = 0
    for i, answer in enumerate(data['answers']):
        if 'text' not in answer:
            return False, f"Antwort {i}: Fehlendes Feld 'text'"
        if 'correct' not in answer:
            return False, f"Antwort {i}: Fehlendes Feld 'correct'"
        if not isinstance(answer['correct'], bool):
            return False, f"Antwort {i}: 'correct' muss boolean sein"
        if answer['correct']:
            correct_count += 1

    if correct_count != 1:
        return False, f"Erwarte genau 1 richtige Antwort, gefunden: {correct_count}"

    return True, "OK"


def save_h5p(data: dict, filename: str):
    """Speichert H5P-Datei."""
    import zipfile

    output_file = OUTPUT_DIR / filename

    # Minimale H5P-Struktur für content.json
    content_json = {
        "question": data['question'],
        "answers": data['answers']
    }

    # Optionale Felder übernehmen falls vorhanden
    if 'behaviour' in data:
        content_json['behaviour'] = data['behaviour']
    else:
        content_json['behaviour'] = {"singleAnswer": True}

    if 'overallFeedback' in data:
        content_json['overallFeedback'] = data['overallFeedback']
    else:
        content_json['overallFeedback'] = [{"from": 0, "to": 100, "text": ""}]

    h5p_json = {
        "title": "Multiple Choice",
        "language": "de",
        "mainLibrary": "H5P.MultiChoice",
        "embedTypes": ["iframe"],
        "license": "U",
        "preloadedDependencies": [
            {
                "machineName": "H5P.MultiChoice",
                "majorVersion": "1",
                "minorVersion": "16"
            }
        ]
    }

    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as h5p:
        h5p.writestr("content/content.json", json.dumps(content_json, ensure_ascii=False, indent=2))
        h5p.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False, indent=2))

    print(f"H5P gespeichert unter: {output_file.resolve()}")


# --------------------------------------
# Hauptfunktion
# --------------------------------------

def generate_h5p(question: str):
    """Generiert H5P-Frage aus einer Eingabe."""

    raw = model_answer(question)
    extracted = extract_json(raw)

    if extracted and "question" in extracted:
        print("✅ JSON erfolgreich extrahiert!")
    else:
        print(f"❌ JSON-Struktur ungültig. Gefunden: {list(extracted.keys()) if extracted else 'Nichts'}")

    print(f"\n{'=' * 80}")
    print(f"🔹 FRAGE: {question}")
    print(f"{'=' * 80}")

    # Modellantwort

    print(f"\n Rohe Antwort:")
    print(f"{'-' * 80}")
    print(raw[:500] + ("..." if len(raw) > 500 else ""))
    print(f"{'-' * 80}")

    # JSON extrahieren


    if extracted is None:
        print("\n Konnte kein valides JSON extrahieren.")
        print(" Tipp: Modell braucht evtl. mehr Training-Epochen")
        return

    print(f"\n JSON erfolgreich extrahiert!")

    # Validierung
    valid, error = validate_h5p_structure(extracted)

    if not valid:
        print(f"\nJSON-Struktur ungültig: {error}")
        print(f"\n Extrahiertes JSON:")
        print(json.dumps(extracted, indent=2, ensure_ascii=False))
        return

    print(f"\n H5P-Struktur valide!")

    # Schön formatiert ausgeben
    print(f"\n Generierte Frage:")
    print(f"{'-' * 80}")
    print(f"Frage: {extracted['question']}")
    print(f"\nAntworten:")
    for i, answer in enumerate(extracted['answers'], 1):
        marker = "✓" if answer['correct'] else " "
        print(f"  [{marker}] {i}. {answer['text']}")
    print(f"{'-' * 80}")

    # Speichern
    save_h5p(extracted, "generated_mc.h5p")

    print(f"\n{'=' * 80}")
    print(" ERFOLGREICH GENERIERT!")
    print(f"{'=' * 80}")


# --------------------------------------
# AUSFÜHRUNG
# --------------------------------------
if __name__ == "__main__":
    # Test mit verschiedenen Fragen
    question =  "Was ist Phishing?"

    generate_h5p(question)
