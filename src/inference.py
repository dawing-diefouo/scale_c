import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.h5p_validator import H5PValidator

# --------------------------------------
# Modellpfad
# --------------------------------------
MODEL_PATH = r"C:\Users\dawin\OneDrive\Documents\Semester_1\Projekt2\scale_c\outputs\final_model_cpu"

print(f"🧠 Lade Modell aus: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to("cpu")
model.eval()

# Speicherordner für erzeugte H5P-Dateien
OUTPUT_DIR = Path("data/h5p")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# --------------------------------------
# Hilfsfunktionen
# --------------------------------------

def build_prompt(question: str) -> str:
    """
    Baut das Chat-Prompt so, wie es im Training genutzt wurde.
    STRICT MODE: Das Modell MUSS valides JSON schreiben.
    """
    system_message = (
        "Du bist ein H5P-Content-Generator. "
        "Erstelle IMMER eine vollständig valide H5P content.json für Multiple-Choice. "
        "Antworte ausschließlich mit JSON ohne Erklärungen."
    )

    prompt = (
        f"<|system|>\n{system_message}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )

    return prompt


def model_answer(question: str) -> str:
    """ Ruft das Modell im STRICT MODE auf. """
    prompt = build_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def extract_json(raw_text: str) -> str | None:
    """ Extrahiert den JSON-Teil aus der Modellantwort. """
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        return raw_text[start:end]
    except:
        return None


def save_h5p(json_text: str, filename: str):
    """ Speichert valides JSON als content.json in einer H5P-Datei. """
    import zipfile

    output_file = OUTPUT_DIR / filename

    h5p_json = {
        "title": "Generated H5P Content",
        "mainLibrary": "H5P.MultiChoice",
        "language": "en",
        "preloadedDependencies": [
            {"machineName": "H5P.MultiChoice", "majorVersion": 1, "minorVersion": 14},
            {"machineName": "H5P.Question", "majorVersion": 1, "minorVersion": 4}
        ]
    }

    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as h5p:
        h5p.writestr("content/content.json", json_text)
        h5p.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False, indent=2))

    print(f"🎉 H5P gespeichert unter: {output_file.resolve()}")


# --------------------------------------
# Hauptfunktion
# --------------------------------------

def generate_h5p(question: str):
    print(f"\n🔹 Frage: {question}")

    # Modellantwort
    raw = model_answer(question)
    extracted = extract_json(raw)

    if extracted is None:
        print("❌ Konnte kein JSON extrahieren.")
        print("Antwort:", raw)
        return

    # STRICT MODE VALIDIERUNG
    ok, error, data = H5PValidator.validate_multiple_choice(extracted)

    if not ok:
        print("❌ Ungültiges JSON:", error)
        print("Antwort:", extracted)
        return

    print("✓ JSON valide")
    save_h5p(extracted, "generated_mc.h5p")


# --------------------------------------
# AUSFÜHRUNG
# --------------------------------------
if __name__ == "__main__":
    frage = "Erstelle eine Multiple-Choice-Frage über Phishing."
    generate_h5p(frage)
