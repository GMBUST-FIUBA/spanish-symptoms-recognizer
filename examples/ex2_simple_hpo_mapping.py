from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

if __name__ == "__main__":
    recognizer = PhenotypesRecognizer(ontology="hpo")
    # Comments about the codes
    # - Fever: HP:0001945
    # - Nausea and vomiting: HP:0002017
    # - Nausea (ideally this one appears): HP:0002018
    possible_expected_results = {"fiebre" : ["HP:0001945"], "náuseas" : ["HP:0002017", "HP:0002018"]}

    print("-------------------------------")
    print(f"Se espera:")
    for (phenotype, codes) in possible_expected_results.items():
        joined_codes = ", ".join(codes)
        print(f" - {phenotype} => {joined_codes}")

    # Results
    results = recognizer.map(possible_expected_results.keys())

    print("-------------------------------")
    print(f"Resultados:")
    for hpo_code in results:
        print(f" - {hpo_code}")
    print("-------------------------------")