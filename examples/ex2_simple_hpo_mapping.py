from symptoms_recognizer.symptom_recognizer import SymptomRecognizer

if __name__ == "__main__":
    recognizer = SymptomRecognizer()
    expected_results = {"Fiebre" : "HP:0001945", "Náusea" : "HP:0002018"}

    print("-------------------------------")
    print(f"Se espera:")
    for (phenotype, code) in expected_results.items():
        print(f" - {phenotype} => {code}")

    # Results
    results = recognizer.map(expected_results.keys())

    print("-------------------------------")
    print(f"Resultados:")
    for hpo_code in results:
        print(f" - {hpo_code}")
    print("-------------------------------")