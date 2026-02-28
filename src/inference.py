# -*- coding: utf-8 -*-

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "final_model_gpu-mistral"

# ------------- CONFIG -------------
MODE = "lora"  # "base" oder "lora"
MERGE = False  # True nur wenn du ein merged Modell willst (Deployment-Test)
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# ----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Mode: {MODE} | Merge: {MERGE}")
print(f"Base: {base_model_name}")
if MODE == "lora":
    print(f"Adapter: {ADAPTER_PATH}")

# Tokenizer IMMER vom Basismodell
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Basismodell laden
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=dtype,
    device_map=None,          # bewusst: wir steuern via .to(device)
    trust_remote_code=True
).to(device)
base_model.eval()
base_model.config.pad_token_id = tokenizer.pad_token_id

# Modell je nach Mode
if MODE == "base":
    model = base_model
else:
    # LoRA Adapter dazu
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    if MERGE:
        model = model.merge_and_unload()  # danach ist es ein "normales" Model ohne PEFT
    model.to(device)
    model.eval()

print("Modell geladen.")
print(f"Device: {device}")

model.to(device)


print("Modell mit LoRA-Adapter geladen!")
print(f"   Device: {device}")

print(f"🧠 Lade LoRA Adapter…")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

# Merge LoRA → zwingt, dass LoRA wirklich angewendet wird
model = model.merge_and_unload()
model.eval()
model.to("cpu")

print("👉 LoRA geladen & gemerged (Inference nutzt Feintuning)")


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


def next_content_filename() -> str:
    """Erzeugt den n?chsten content_XXX.json Namen."""
    pattern = re.compile(r"^content_(\d{3})\.json$")
    max_idx = 0

    if OUTPUT_DIR.exists():
        for path in OUTPUT_DIR.iterdir():
            match = pattern.match(path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

    return f"content_{max_idx + 1:03d}.json"

def generate_content(question: str, show_output: bool = True):
    """Generiert Content aus einer Eingabe."""

    raw = model_answer(question)
    extracted = extract_json(raw)
    payload = extracted if extracted is not None else raw
    if show_output:
        preview = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        print("\nVorschau (gekuerzt):")
        print(preview[:500] + ("..." if len(preview) > 500 else ""))
    output_file = OUTPUT_DIR / next_content_filename()
    with open(output_file, "w", encoding="utf-8") as f:
        if isinstance(payload, dict):
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(payload))
    if show_output:
        print(f"{output_file.name} gespeichert")



# =============================================================================
# 7. AUSFÜHRUNG
# =============================================================================
if __name__ == "__main__":
    # Test mit mehreren Fragen aus Datei (eine Frage pro Zeile)
    questions_path = Path("data/processed/test_questions.txt")
    if not questions_path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {questions_path.resolve()}")

    questions = [
        line.strip()
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for question in questions:
        generate_content(question, show_output=False)
    print(f"{len(questions)} Inhalte gespeichert")
