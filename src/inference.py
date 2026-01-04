import json
import torch
import re
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "h5p"
DATA_DIR.mkdir(parents=True, exist_ok=True)


from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from src.h5p_validator import H5PValidator


# =============================================================================
# 1. MODEL-PFAD
# =============================================================================
MODEL_PATH = r"C:\Users\dawin\OneDrive\Documents\Semester_1\Projekt2\scale_c\outputs\final_model_cpu"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"🧠 Lade Basismodell…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

print(f"🧠 Lade LoRA Adapter…")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

# Merge LoRA → zwingt, dass LoRA wirklich angewendet wird
model = model.merge_and_unload()
model.eval()
model.to("cpu")

print("👉 LoRA geladen & gemerged (Inference nutzt Feintuning)")


# =============================================================================
# 2. PROMPT GENERATOR – IDENTISCH ZUM TRAINING!
# =============================================================================
def build_prompt(question: str) -> str:
    """
    Der Prompt MUSS 1:1 das SFT-Trainingsformat nachbilden.
    """

    instruction_text = (
        "Erstelle eine H5P-Multiple-Choice-Frage mit 4 Antwortmöglichkeiten "
        "und 1 richtigen Antwort(en), basierend auf folgender Frage: "
        f"'{question}'. "
        "Antworte ausschließlich als einzelnes gültiges JSON-Objekt "
        "im korrekten H5P-MultipleChoice-Format. Keine Erklärungen."
    )

    return (
        f"<|system|>\n"
        f"Du bist ein H5P-Content-Generator.</s>\n"
        f"<|user|>\n{instruction_text}</s>\n"
        f"<|assistant|>\n"
    )


# =============================================================================
# 3. JSON EXTRACTOR – nimmt NUR das erste vollständige JSON
# =============================================================================
def extract_first_json(text: str) -> str | None:
    """
    Findet das erste vollständige JSON-Objekt im Modelltext.
    """
    pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        return None

    first = matches[0]

    try:
        json.loads(first)
        return first
    except:
        return None


# =============================================================================
# 4. ANTWORT GENERIEREN
# =============================================================================
def model_answer(question: str) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=758,
            do_sample=False,    # deterministisch
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0
        )

    return tokenizer.decode(output[0], skip_special_tokens=False)


# =============================================================================
# 5. H5P SPEICHERN
# =============================================================================

def save_h5p(json_text: str, filename: str):
    import zipfile

    output_file = DATA_DIR / filename

    h5p_json = {
        "title": "Multiple Choice",
        "language": "en",
        "mainLibrary": "H5P.MultiChoice",
        "embedTypes": [
            "iframe"
        ],
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
        h5p.writestr("content/content.json", json_text)
        h5p.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False, indent=2))

    print(f"🎉 H5P gespeichert unter: {output_file.resolve()}")


# =============================================================================
# 6. HAUPTFUNKTION
# =============================================================================
def generate_h5p(question: str):
    print(f"\n🔹 Frage: {question}")

    raw = model_answer(question)
    print("🔎 Modellrohantwort:\n", raw)

    extracted = extract_first_json(raw)

    if extracted is None:
        print("❌ Konnte kein gültiges JSON extrahieren.")
        return

    ok, err, parsed = H5PValidator.validate_multiple_choice(extracted)

    if not ok:
        print(f"❌ JSON ungültig: {err}")
        print("Antwort:", extracted)
        return

    print("✅ JSON gültig!")
    save_h5p(extracted, "generated_mc.h5p")


# =============================================================================
# 7. AUSFÜHRUNG
# =============================================================================
if __name__ == "__main__":
    frage = "Was ist Quid-Pro-Quo-Phishing?"
    generate_h5p(frage)
