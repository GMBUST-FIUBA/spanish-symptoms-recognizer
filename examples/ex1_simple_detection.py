from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

if __name__ == "__main__":
    recognizer = PhenotypesRecognizer(
        ontology="hpo", phenotypes_model_type="api",
        ner_api_provider="openai", ner_api_model_name="gpt-4o-mini",
    )
    text = "El paciente tiene mareos."
    expected_results = ["mareos"]

    print("-------------------------------")
    print(f"En el texto: {text}\n")
    print(f"Se espera encontrar:")
    for result in expected_results:
        print(f" - {result}")

    # Results
    results = recognizer.recognize(text)

    print("-------------------------------")
    print(f"Síntomas encontrados:")
    for result in results:
        print(f" - {result}")
    print("-------------------------------")