from symptoms_recognizer.symptom_recognizer import SymptomRecognizer

if __name__ == "__main__":
    recognizer = SymptomRecognizer()
    text = "El paciente tiene mareos."

    results = recognizer.recognize(text)

    print(f"Resultados: {results}")