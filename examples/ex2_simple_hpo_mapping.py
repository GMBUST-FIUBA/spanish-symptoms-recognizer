from symptoms_recognizer.symptom_recognizer import SymptomRecognizer

if __name__ == "__main__":
    recognizer = SymptomRecognizer()
    symptoms_list = ["mareos", "náuseas"]

    results = recognizer.map(symptoms_list)

    print(f"Resultados: {results}")