from pathlib import Path
from math import inf
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
import torch

# Get current directory
CURRENT_DIR = Path(__file__).parent.resolve()

# Accepted Ontologies
HPO_ONTOLOGY_CODE = "hpo"

ACCEPTED_ONTOLOGIES = { HPO_ONTOLOGY_CODE }

# Accepted ontologies files

HPO_FILE_RELATIVE_PATH = "hpo/hpo-tokens.pt"
HPO_ABSOLUTE_INPUT_FILE_PATH = os.path.join(CURRENT_DIR, HPO_FILE_RELATIVE_PATH)

ACCEPTED_ONTOLOGIES_FILES = {
    HPO_ONTOLOGY_CODE : HPO_ABSOLUTE_INPUT_FILE_PATH
}

# Default location for entity linking model
LOCAL_MODEL_RELATIVE_PATH = "semantic_model"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)


# Minimun distance between vectors
MIN_DISTANCE_VECTORS = 0.2


class PhenotypeOntologyMapper:
    def __init__(self, model_path = None, tokenizer_path = None, ontology = None, ontology_file_path=None):
        # Get ontology
        if ontology in ACCEPTED_ONTOLOGIES:
            self.mapped_ontology = ontology
            self.ontology_file_path = ACCEPTED_ONTOLOGIES_FILES[ontology]

        elif ontology and not ontology_file_path:
            raise Exception("TODO!")

        else:
            raise Exception("No ontology defined.")

        # Get models
        if not model_path:
            self.model = self._get_el_model(DEFAULT_LOCAL_MODEL_PATH)
        else:
            self.model = self._get_el_model(model_path)

        if not tokenizer_path:
            self.tokenizer = self._get_el_tokenizer(DEFAULT_LOCAL_MODEL_PATH)
        else:
            self.tokenizer = self._get_el_tokenizer(tokenizer_path)

    def _get_el_model(self, path):
        return AutoModel.from_pretrained(path)

    def _get_el_tokenizer(self, path):
        return AutoTokenizer.from_pretrained(path)

    def map_phenotypes(self, phenotypes_list: list[str]):
        # Encode symptoms
        encoded_phenotypes_list = self._get_encoded_phenotypes_list(phenotypes_list)

        # Open file of embedings
        with open(self.ontology_file_path, "rb") as input_file:
            # For every codes batch
            hpo_codes_for_phenotypes = [("", inf) for _ in range(len(phenotypes_list))]
            hpo_codes_batch = self._get_codes_batch(input_file)

            # For every code in the batch
            for hpo_code in hpo_codes_batch:

                # For every symptom in the list
                for pos, hpo_info in enumerate(hpo_codes_for_phenotypes):
                    # Calculate distance between vectors
                    vectors_distance = self._calculate_distance(encoded_phenotypes_list[pos], hpo_codes_batch[hpo_code])
                    # Compare with old distance
                    if hpo_info[1] > vectors_distance:
                        hpo_codes_for_phenotypes[pos] = (hpo_code, vectors_distance)

        # Apply similarity minimum
        phenotypes_mapped = []

        for hpo_code, similarity in hpo_codes_for_phenotypes:
            if similarity <= MIN_DISTANCE_VECTORS:
                phenotypes_mapped.append(hpo_code)
            else:
                phenotypes_mapped.append("None")

        return phenotypes_mapped

    def _get_encoded_phenotypes_list(self, phenotypes_list: list[str]):
        encoded_phenotypes_list = []
        with torch.no_grad():
            for symptom in phenotypes_list:
                tokenized_symptom = self.tokenizer(symptom, return_tensors="pt", padding=True, truncation=True, max_length=512)
                model_output = self.model(**tokenized_symptom)
                cls_embedding = model_output.last_hidden_state[0, 0, :]
                symptom_embedding = F.normalize(cls_embedding.unsqueeze(0), p=2, dim=1).squeeze().numpy()

                encoded_phenotypes_list.append(symptom_embedding)

        return encoded_phenotypes_list

    def _get_codes_batch(self, input_file):
        return torch.load(input_file, map_location=torch.device('cpu'))

    def _calculate_distance(self, vector1, vector2):
        return distance.cosine(vector1, vector2)