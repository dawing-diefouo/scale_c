import json
from typing import Optional, Dict, List


class H5PValidator:
    """Validiert H5P-MultipleChoice-JSON-Strukturen"""

    @staticmethod
    def validate_multiple_choice(h5p_json: str) -> tuple[bool, Optional[str], Optional[Dict]]:

        # 1. JSON parsen
        try:
            data = json.loads(h5p_json)
        except json.JSONDecodeError as e:
            return False, f"Invalides JSON: {str(e)}", None

        # 2. Basis-Struktur prüfen
        if not isinstance(data, dict):
            return False, "H5P muss ein JSON-Objekt sein", None

        # 3. Pflichtfelder prüfen
        required_fields = ["question", "answers"]
        for field in required_fields:
            if field not in data:
                return False, f"Fehlendes Pflichtfeld: {field}", None

        # 4. Question validieren
        if not isinstance(data["question"], str) or not data["question"].strip():
            return False, "Feld 'question' muss ein nicht-leerer String sein", None

        # 5. Answers validieren
        if not isinstance(data["answers"], list):
            return False, "Feld 'answers' muss eine Liste sein", None

        if len(data["answers"]) < 2:
            return False, "Mindestens 2 Antwortmöglichkeiten erforderlich", None

        # 6. Jede Antwort validieren
        correct_count = 0
        for i, answer in enumerate(data["answers"]):
            # Muss ein Dict sein
            if not isinstance(answer, dict):
                return False, f"Antwort {i + 1} muss ein Objekt sein", None

            # Text erforderlich
            if "text" not in answer:
                return False, f"Antwort {i + 1}: Fehlendes Feld 'text'", None

            if not isinstance(answer["text"], str) or not answer["text"].strip():
                return False, f"Antwort {i + 1}: 'text' muss ein nicht-leerer String sein", None

            # Correct erforderlich
            if "correct" not in answer:
                return False, f"Antwort {i + 1}: Fehlendes Feld 'correct'", None

            if not isinstance(answer["correct"], bool):
                return False, f"Antwort {i + 1}: 'correct' muss true oder false sein", None

            if answer["correct"]:
                correct_count += 1

        # 7. Mindestens eine richtige Antwort
        if correct_count == 0:
            return False, "Mindestens eine Antwort muss als 'correct': true markiert sein", None

        # 8. Behaviour validieren (optional, aber empfohlen)
        if "behaviour" in data:
            if not isinstance(data["behaviour"], dict):
                return False, "Feld 'behaviour' muss ein Objekt sein", None

            if "singleAnswer" in data["behaviour"]:
                if not isinstance(data["behaviour"]["singleAnswer"], bool):
                    return False, "'singleAnswer' muss true oder false sein", None

                # Bei singleAnswer: true darf nur eine richtige Antwort existieren
                if data["behaviour"]["singleAnswer"] and correct_count > 1:
                    return False, f"Bei 'singleAnswer': true ist nur 1 richtige Antwort erlaubt, gefunden: {correct_count}", None

        # 9. OverallFeedback validieren (optional)
        if "overallFeedback" in data:
            if not isinstance(data["overallFeedback"], list):
                return False, "Feld 'overallFeedback' muss eine Liste sein", None

        # ✅ Alles valide!
        return True, None, data

    @staticmethod
    def get_validation_summary(h5p_json: str) -> Dict:

        is_valid, error, data = H5PValidator.validate_multiple_choice(h5p_json)

        summary = {
            "is_valid": is_valid,
            "error": error,
            "stats": None
        }

        if data:
            summary["stats"] = {
                "question_length": len(data["question"]),
                "answer_count": len(data["answers"]),
                "correct_count": sum(1 for a in data["answers"] if a.get("correct", False)),
                "has_behaviour": "behaviour" in data,
                "has_feedback": "overallFeedback" in data,
                "single_answer_mode": data.get("behaviour", {}).get("singleAnswer", False)
            }

        return summary

    @staticmethod
    def fix_common_issues(h5p_json: str) -> tuple[bool, str]:

        try:
            data = json.loads(h5p_json)
        except:
            return False, h5p_json

        fixed = False

        # 1. Füge behaviour hinzu falls fehlend
        if "behaviour" not in data:
            # Prüfe ob mehrere richtige Antworten
            correct_count = sum(1 for a in data.get("answers", []) if a.get("correct", False))
            data["behaviour"] = {"singleAnswer": correct_count == 1}
            fixed = True

        # 2. Füge overallFeedback hinzu falls fehlend
        if "overallFeedback" not in data:
            data["overallFeedback"] = [{"from": 0, "to": 100, "text": ""}]
            fixed = True

        # 3. Korrigiere String-Booleans in answers
        if "answers" in data:
            for answer in data["answers"]:
                if "correct" in answer:
                    if answer["correct"] == "true":
                        answer["correct"] = True
                        fixed = True
                    elif answer["correct"] == "false":
                        answer["correct"] = False
                        fixed = True

        # 4. Trimme Whitespace
        if "question" in data and isinstance(data["question"], str):
            trimmed = data["question"].strip()
            if trimmed != data["question"]:
                data["question"] = trimmed
                fixed = True

        if "answers" in data:
            for answer in data["answers"]:
                if "text" in answer and isinstance(answer["text"], str):
                    trimmed = answer["text"].strip()
                    if trimmed != answer["text"]:
                        answer["text"] = trimmed
                        fixed = True

        return fixed, json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def create_example_multiple_choice() -> str:
        """Gibt ein Beispiel für valides MultipleChoice-JSON zurück"""
        example = {
            "question": "Was ist die Hauptstadt von Deutschland?",
            "answers": [
                {"text": "Berlin", "correct": True},
                {"text": "München", "correct": False},
                {"text": "Hamburg", "correct": False},
                {"text": "Köln", "correct": False}
            ],
            "behaviour": {
                "singleAnswer": True
            },
            "overallFeedback": [
                {"from": 0, "to": 100, "text": ""}
            ]
        }
        return json.dumps(example, ensure_ascii=False, indent=2)
