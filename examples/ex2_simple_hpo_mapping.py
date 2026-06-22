from symptoms_recognizer.symptom_recognizer import SymptomRecognizer

if __name__ == "__main__":
    recognizer = SymptomRecognizer()
    expected_results = {"Fiebre" : "HP:0001945", "Náusea" : "HP:0002018"}

    results = recognizer.map(expected_results.keys())

    print("-------------------------------")
    print(f"Se espera:")
    print(f" - Fiebre => HP:0001945")
    print(f" - Náusea => HP:0002018")
    print("-------------------------------")
    print(f"Resultados:")
    for hpo_code in results:
        print(f" - {hpo_code}")