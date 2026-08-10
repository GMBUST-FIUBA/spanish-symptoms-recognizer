from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

if __name__ == "__main__":
    # Read clinical history
    with open("./examples/example_docs/ex3_example_doc.txt") as document:
        clinical_history = document.read()

    # Get phenotypes recognizer
    recognizer = PhenotypesRecognizer(ontology="hpo")

    # Expected results:
    expected_results = {
        "HP:0002315" : "Cefalea / Dolor de cabeza",
        "HP:0002013" : "Vómitos",
        "HP:0002321" : "Mareos / Vértigo",
        "HP:0002354" : "Amnesia / Pérdida de memoria",
        "HP:0001289" : "Desorientación / Confusión",
        "HP:0000822" : "Presión arterial elevada",
        "HP:0001017" : "Palidez",
        "HP:0002066" : "Ataxia / Inestabilidad",
    }

    print("--------------------------------\n")
    print("Se espera:")
    for code, explanation in expected_results.items():
        print(f" - {code} : {explanation}")
    print()
    print("--------------------------------\n")
    print("Se obtuvo:")

    results = recognizer.scan(clinical_history, only_results=False)

    for result in results:
        print(f" - {result[1]} : {result[0]}")

    print("--------------------------------")