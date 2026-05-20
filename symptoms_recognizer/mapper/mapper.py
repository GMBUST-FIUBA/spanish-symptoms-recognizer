def get_codes_batch():
    raise Exception("TODO: HPO batches generator")

def get_symptoms(symptoms_list: list[str]):

    # For every codes batch
    for hpo_codes_batch in get_codes_batch():

        # For every symptom in the list
        for symptom in symptoms_list:
            pass