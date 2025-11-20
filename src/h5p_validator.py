import json
from typing import Optional, Dict


class H5PValidator:
    """Validiert H5P-MultipleChoice-JSON-Strukturen im STRICT MODE."""

    @staticmethod
    def validate_multiple_choice(h5p_json: str) -> tuple[bool, Optional[str], Optional[Dict]]:

        # 1. JSON parsen
        try:
            data = json.loads(h5p_json)
        except json.JSONDecodeError as e:
            return False, f"Invalides JSON: {str(e)}", None

        # 2. Basisstruktur prüfen
        if not isinstance(data, dict):
            return False, "H5P muss ein JSON-Objekt sein", None

        # 3. Pflichtfelder prüfen
        required_fields = ["question", "answers"]
        for field in required_fields:
            if field not in data:
                return False, f"Fehlendes Pflichtfeld: {field}", None

        # 4. Frage prüfen
        if not isinstance(data["question"], str) or not data["question"].strip():
            return False, "Feld 'question' muss ein nicht-leerer String sein", None

        # 5. Antworten prüfen
        if not isinstance(data["answers"], list):
            return False, "Feld 'answers' muss eine Liste sein", None

        if len(data["answers"]) < 2:
            return False, "Mindestens 2 Antwortmöglichkeiten erforderlich", None

        # 6. Jede Antwort prüfen
        correct_count = 0
        for i, answer in enumerate(data["answers"]):
            if not isinstance(answer, dict):
                return False, f"Antwort {i+1} muss ein Objekt sein", None

            if "text" not in answer:
                return False, f"Antwort {i+1}: Fehlendes Feld 'text'", None

            if not isinstance(answer["text"], str) or not answer["text"].strip():
                return False, f"Antwort {i+1}: 'text' muss ein nicht-leerer String sein", None

            if "correct" not in answer:
                return False, f"Antwort {i+1}: Fehlendes Feld 'correct'", None

            if not isinstance(answer["correct"], bool):
                return False, f"Antwort {i+1}: 'correct' muss true oder false sein", None

            if answer["correct"]:
                correct_count += 1

        if correct_count == 0:
            return False, "Mindestens eine Antwort muss als 'correct': true markiert sein", None

        # 7. Optional: behaviour prüfen
        if "behaviour" in data:
            if not isinstance(data["behaviour"], dict):
                return False, "Feld 'behaviour' muss ein Objekt sein", None

            if "singleAnswer" in data["behaviour"]:
                if not isinstance(data["behaviour"]["singleAnswer"], bool):
                    return False, "'singleAnswer' muss true oder false sein", None

                # Wenn singleAnswer=true → nur 1 richtige Antwort erlaubt
                if data["behaviour"]["singleAnswer"] and correct_count != 1:
                    return False, f"Bei 'singleAnswer': true ist genau 1 richtige Antwort erlaubt, gefunden: {correct_count}", None

        # 8. Optional: overallFeedback prüfen
        if "overallFeedback" in data:
            if not isinstance(data["overallFeedback"], list):
                return False, "Feld 'overallFeedback' muss eine Liste sein", None

        # Wenn alles gültig ist
        return True, None, data

