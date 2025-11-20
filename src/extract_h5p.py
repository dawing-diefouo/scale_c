import zipfile
import json
from pathlib import Path
import os

# === Pfade ===
INPUT_DIR = Path("data/raw")                # H5P-Dateien-Ordner
OUTPUT_DIR = Path("data/processed")       # Zielordner
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_data.jsonl")


def extract_h5p_content_json(h5p_path):
    """Extrahiert content.json aus einer H5P-Datei (ZIP)."""
    with zipfile.ZipFile(h5p_path, 'r') as z:
        # Suchen nach content/content.json
        for name in z.namelist():
            if name.endswith("content.json"):
                with z.open(name) as f:
                    return json.load(f)
    return None


def generate_instruction(data):
    """Erzeugt automatisch eine sinnvolle Instruction aus content.json."""

    question = data.get("question", "").strip()

    # Wenn keine Frage gefunden wurde, generische Instruktion nehmen
    if not question:
        return "Erstelle eine H5P-Multiple-Choice-Frage basierend auf den folgenden Daten."

    # Spezialfall: MC-Fragen
    answers = data.get("answers", [])
    num_answers = len(answers)
    num_correct = sum(1 for a in answers if a.get("correct"))

    return (
        f"Erstelle eine H5P-Multiple-Choice-Frage mit {num_answers} Antwortmöglichkeiten "
        f"und {num_correct} richtigen Antwort(en), basierend auf folgender Frage: '{question}'."
    )


def convert_h5p_folder_to_instruction_pairs(input_dir, output_file):
    all_pairs = 0

    with open(output_file, "w", encoding="utf-8") as outfile:

        for filename in os.listdir(input_dir):
            if not filename.endswith(".h5p"):
                continue

            path = os.path.join(input_dir, filename)
            print(f"Verarbeite: {filename}")

            content_json = extract_h5p_content_json(path)
            if content_json is None:
                print(f"content.json nicht gefunden in {filename}")
                continue

            instruction = generate_instruction(content_json)

            # OUTPUT MUSS EIN STRING SEIN (für Finetuning!)
            output_json_string = json.dumps(content_json, ensure_ascii=False)

            record = {
                "instruction": instruction,
                "output": output_json_string
            }

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_pairs += 1

    print(f"\nFertig! {all_pairs} Instruction-Paare wurden erzeugt.")
    print(f"Gespeichert in: {output_file}")
