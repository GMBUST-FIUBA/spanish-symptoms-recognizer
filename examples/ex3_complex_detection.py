from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

if __name__ == "__main__":
    # Read clinical history
    with open("./examples/example_docs/example_doc_1.txt") as document:
        clinical_history = document.read()

    # Get phenotypes recognizer
    recognizer = PhenotypesRecognizer(ontology="hpo", text_parser="sections-sentences-parser")

    # Expected results:
    expected_results = {
        "HP:0002315" : "Cefalea",
        "HP:0001289" : "Confusión",
        "HP:0011422" : "Amnesia anterógrada",
        "HP:0011423" : "Amnesia retrógrada",
        "HP:0002321" : "Vértigo",
        "HP:0001251" : "Ataxia",
        "HP:0002076" : "Migraña",
        "HP:0002013" : "Vómitos",
        "HP:0002018" : "Náuseas",
        "HP:0000822" : "Hipertensión arterial",
        "HP:0000980" : "Palidez",
        "HP:0001944" : "Deshidratación",
    }

    print("--------------------------------\n")
    print("Se espera (o lo más parecido a):")
    for code, explanation in expected_results.items():
        print(f" - {code} : {explanation}")
    print()
    print("--------------------------------\n")
    print("Se obtuvo:")

    results = recognizer.recognize(clinical_history)

    for result in results:
        print(f" - {result}")

    print("--------------------------------")