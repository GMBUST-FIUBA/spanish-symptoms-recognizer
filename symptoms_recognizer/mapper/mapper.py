from transformers import AutoTokenizer, AutoModel
from math import inf

import torch

# Embeddings generator
## Local path
LOCAL_TOKENIZER_PATH = "../base_model/embedding/roberta-base-biomedical-clinical-es"
LOCAL_MODEL_PATH = "../output/model"

def _get_encoded_symptoms_list(symptoms_list: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
    encoded_symptoms_list = []
    with torch.no_grad():
        for symptom in symptoms_list:
            tokenized_symptom = tokenizer(symptom, return_tensors="pt", padding=True, truncation=True, max_length=512)
            model_output = model(**tokenized_symptom)
            symptom_embedding = model_output.last_hidden_state[0, 0, :].numpy()

            encoded_symptoms_list.append(symptom_embedding)
    
    return encoded_symptoms_list

def get_codes_batch():
    raise Exception("TODO: HPO batches generator")

def calculate_distance(vector1, vector2):
    return 0

def get_symptoms(symptoms_list: list[str]):
    # Encode symptoms
    encoded_symptoms_list = _get_encoded_symptoms_list(symptoms_list)

    # For every codes batch
    hpo_codes_for_symptoms = [("", inf) for _ in range(len(symptoms_list))]
    for hpo_codes_batch in get_codes_batch():

        # For every code in the batch
        for hpo_code in hpo_codes_batch:

            # For every symptom in the list
            for pos, hpo_info in enumerate(hpo_codes_for_symptoms):
                # Calculate distance between vectors
                vectors_distance = calculate_distance(encoded_symptoms_list[pos], hpo_codes_batch[hpo_code])
                # Compare with old distance
                if hpo_info[1] > vectors_distance:
                    hpo_codes_for_symptoms[pos] = (hpo_code, vectors_distance)

    return [hpo_code for hpo_code, _ in hpo_codes_for_symptoms]