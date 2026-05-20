from pathlib import Path
from math import inf
from scipy.spatial import distance

import os
import torch

# Codes batch size
HPO_CODES_BATCH_SIZE = 500

# Input file
CURRENT_DIR = Path(__file__).parent.resolve()
INPUT_FILE_RELATIVE_PATH = "hpo/hpo-tokens.pt"
ABSOLUTE_INPUT_FILE_PATH = os.path.join(CURRENT_DIR, INPUT_FILE_RELATIVE_PATH)


def _get_encoded_symptoms_list(symptoms_list: list[str], tokenizer, model):
    encoded_symptoms_list = []
    with torch.no_grad():
        for symptom in symptoms_list:
            print("Nuevo síntoma a codificar")
            tokenized_symptom = tokenizer(symptom, return_tensors="pt", padding=True, truncation=True, max_length=512)
            model_output = model.base_model(**tokenized_symptom)
            symptom_embedding = model_output.last_hidden_state[0, 0, :].numpy()

            encoded_symptoms_list.append(symptom_embedding)
    
    return encoded_symptoms_list

def _get_codes_batch(input_file):
    return torch.load(input_file)

def _calculate_similarity(vector1, vector2):
    return distance.cosine(vector1, vector2)

# Mapper between symptoms and HPO codes
def map_symptoms_to_codes(symptoms_list: list[str], tokenizer, model):
    # Encode symptoms
    encoded_symptoms_list = _get_encoded_symptoms_list(symptoms_list, tokenizer, model)

    # Open file of embedings
    with open(ABSOLUTE_INPUT_FILE_PATH, "rb") as input_file:
        # For every codes batch
        hpo_codes_for_symptoms = [("", inf) for _ in range(len(symptoms_list))]
        hpo_codes_batch = _get_codes_batch(input_file)

        # For every code in the batch
        for hpo_code in hpo_codes_batch:

            # For every symptom in the list
            for pos, hpo_info in enumerate(hpo_codes_for_symptoms):
                # Calculate distance between vectors
                vectors_distance = _calculate_similarity(encoded_symptoms_list[pos], hpo_codes_batch[hpo_code])
                # Compare with old distance
                if hpo_info[1] > vectors_distance:
                    hpo_codes_for_symptoms[pos] = (hpo_code, vectors_distance)

    return [hpo_code for hpo_code, _ in hpo_codes_for_symptoms]