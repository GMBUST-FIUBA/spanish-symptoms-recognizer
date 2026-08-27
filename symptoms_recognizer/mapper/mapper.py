from pathlib import Path
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel

import glob
import torch.nn.functional as F
import os
import torch

# Get current directory
CURRENT_DIR = Path(__file__).parent.resolve()

# Accepted Ontologies
HPO_ONTOLOGY_CODE = "hpo"

ACCEPTED_ONTOLOGIES = { HPO_ONTOLOGY_CODE }

# Accepted ontologies files

HPO_FILE_RELATIVE_PATH = "hpo/hpo_batches"
HPO_ABSOLUTE_INPUT_FILE_PATH = os.path.join(CURRENT_DIR, HPO_FILE_RELATIVE_PATH)

ACCEPTED_ONTOLOGIES_FILES = {
    HPO_ONTOLOGY_CODE : HPO_ABSOLUTE_INPUT_FILE_PATH
}

# Default location for entity linking model
LOCAL_MODEL_RELATIVE_PATH = "semantic_model/clinlinker-kb-gp"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)


# Minimun distance between vectors
MIN_DISTANCE_VECTORS = 0.1


class PhenotypeOntologyMapper:
    def __init__(self, model_path = None, tokenizer_path = None, ontology = None, ontology_file_path=None):
        # Set operations device
        self.pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.model = self._get_el_model(DEFAULT_LOCAL_MODEL_PATH).to(self.pytorch_device)
        else:
            self.model = self._get_el_model(model_path).to(self.pytorch_device)

        if not tokenizer_path:
            self.tokenizer = self._get_el_tokenizer(DEFAULT_LOCAL_MODEL_PATH)
        else:
            self.tokenizer = self._get_el_tokenizer(tokenizer_path)

    def _get_el_model(self, path):
        return AutoModel.from_pretrained(path)

    def _get_el_tokenizer(self, path):
        return AutoTokenizer.from_pretrained(path)

    def map_phenotypes(self, phenotypes_list: list[str]):
        # Encode phenotypes to map
        encoded_phenotypes_matrix = self._get_encoded_phenotypes_list(phenotypes_list)
        total_phenotypes = len(phenotypes_list)

        # Create buffers for mapped phenotypes
        best_similarities = torch.full((total_phenotypes,), float('-inf'), device=self.pytorch_device)
        mapped_phenotypes = ["None"] * total_phenotypes

        # Go over all batches
        for hpo_codes_batch in self._get_codes_batch():
            # Get batch codes
            codes_in_batch = list(hpo_codes_batch.keys())
            vectors_in_batch = torch.stack(list(hpo_codes_batch.values())).to(self.pytorch_device)
            
            # Flatten dimensions
            if vectors_in_batch.dim() == 3:
                vectors_in_batch = vectors_in_batch.squeeze(1)

            similarities = torch.matmul(encoded_phenotypes_matrix, vectors_in_batch.T)
            max_sims, max_indices = torch.max(similarities, dim=1)

            # Update codes
            for i in range(total_phenotypes):
                if max_sims[i] > best_similarities[i]:
                    best_similarities[i] = max_sims[i]
                    mapped_phenotypes[i] = codes_in_batch[max_indices[i]]

        min_required_similarity = 1.0 - MIN_DISTANCE_VECTORS

        for i in range(total_phenotypes):
            if best_similarities[i] < min_required_similarity:
                mapped_phenotypes[i] = "None"

        return mapped_phenotypes

    def _get_codes_batch(self):
        batch_files = glob.glob(os.path.join(self.ontology_file_path, "*.pt"))

        for batch_file in batch_files:
            raw_dict = torch.load(batch_file, map_location=self.pytorch_device)
            yield raw_dict

            del raw_dict
            if self.pytorch_device.type == 'cuda':
                torch.cuda.empty_cache()

    def _calculate_distance(self, vector1, vector2):
        similitud = torch.dot(vector1, vector2).item()
        return 1.0 - similitud

    def _get_encoded_phenotypes_list(self, phenotypes_list: list[str]):
        with torch.no_grad():
            tokenized_symptoms = self.tokenizer(
                phenotypes_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512).to(self.pytorch_device)

            phenotypes_model_output = self.model(**tokenized_symptoms)
            phenotypes_cls_embedding = phenotypes_model_output.last_hidden_state[:, 0, :]
            encoded_phenotypes_list = F.normalize(phenotypes_cls_embedding, p=2, dim=1)

        return encoded_phenotypes_list