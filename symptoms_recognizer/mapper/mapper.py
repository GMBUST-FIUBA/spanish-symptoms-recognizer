from pathlib import Path
from scipy.spatial import distance
from transformers import AutoTokenizer, AutoModel
from google import genai
from openai import OpenAI

import glob
import torch.nn.functional as F
import os
import torch
import heapq
import json
import re

CURRENT_DIR = Path(__file__).parent.resolve()
HPO_ONTOLOGY_CODE = "hpo"
ACCEPTED_ONTOLOGIES = { HPO_ONTOLOGY_CODE }
HPO_FILE_RELATIVE_PATH = "hpo/hpo_batches"
HPO_ABSOLUTE_INPUT_FILE_PATH = os.path.join(CURRENT_DIR, HPO_FILE_RELATIVE_PATH)
ACCEPTED_ONTOLOGIES_FILES = { HPO_ONTOLOGY_CODE : HPO_ABSOLUTE_INPUT_FILE_PATH }

LOCAL_MODEL_RELATIVE_PATH = "semantic_model/clinlinker-kb-gp"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)
MIN_DISTANCE_VECTORS = 0.1

class PhenotypeOntologyMapper:
    def __init__(self, model_path=None, tokenizer_path=None, ontology=None, ontology_file_path=None, 
                 api_provider=None, api_model_name=None, top_k=5):
        
        self.pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.api_provider = api_provider
        self.api_model_name = api_model_name
        self.top_k = top_k
        self.min_required_similarity = 1.0 - MIN_DISTANCE_VECTORS

        if self.api_provider == "gemini": self.client = genai.Client()
        elif self.api_provider == "openai": self.client = OpenAI()
        elif self.api_provider: raise Exception(f"Proveedor no soportado: {self.api_provider}")

        if ontology in ACCEPTED_ONTOLOGIES:
            self.mapped_ontology = ontology
            self.ontology_file_path = ACCEPTED_ONTOLOGIES_FILES[ontology]
        elif ontology and not ontology_file_path: raise Exception("TODO!")
        else: raise Exception("No ontology defined.")

        model_p = model_path or DEFAULT_LOCAL_MODEL_PATH
        tok_p = tokenizer_path or DEFAULT_LOCAL_MODEL_PATH
        self.model = self._get_el_model(model_p).to(self.pytorch_device)
        self.tokenizer = self._get_el_tokenizer(tok_p)

    def _get_el_model(self, path): return AutoModel.from_pretrained(path)
    def _get_el_tokenizer(self, path): return AutoTokenizer.from_pretrained(path)

    def map_phenotypes(self, phenotypes_with_context: list[tuple[str, str]]):
        if not phenotypes_with_context:
            return []

        phenotypes_list = [item[0] for item in phenotypes_with_context]
        
        encoded_phenotypes_matrix = self._get_encoded_phenotypes_list(phenotypes_list)
        total_phenotypes = len(phenotypes_list)

        heaps = [[] for _ in range(total_phenotypes)]

        for hpo_codes_batch in self._get_codes_batch():
            codes_in_batch = list(hpo_codes_batch.keys())

            vectors_in_batch = torch.stack([hpo_codes_batch[c]["vector"] for c in codes_in_batch]).to(self.pytorch_device)
            names_in_batch = [hpo_codes_batch[c]["name"] for c in codes_in_batch]
            
            if vectors_in_batch.dim() == 3:
                vectors_in_batch = vectors_in_batch.squeeze(1)

            similarities = torch.matmul(encoded_phenotypes_matrix, vectors_in_batch.T)

            k_batch = min(self.top_k, similarities.size(1))
            batch_topk_sims, batch_topk_indices = torch.topk(similarities, k_batch, dim=1)

            batch_topk_sims = batch_topk_sims.cpu().tolist()
            batch_topk_indices = batch_topk_indices.cpu().tolist()

            for i in range(total_phenotypes):
                for sim, idx in zip(batch_topk_sims[i], batch_topk_indices[i]):
                    if sim < self.min_required_similarity: continue
                    
                    code = codes_in_batch[idx]
                    name = names_in_batch[idx]
                    
                    if len(heaps[i]) < self.top_k:
                        heapq.heappush(heaps[i], (sim, code, name))
                    else:
                        heapq.heappushpop(heaps[i], (sim, code, name))

        mapped_phenotypes = ["None"] * total_phenotypes
        
        for i in range(total_phenotypes):
            if not heaps[i]: continue
                
            sorted_candidates = sorted(heaps[i], key=lambda x: x[0], reverse=True)

            if not self.api_provider:
                mapped_phenotypes[i] = sorted_candidates[0][1]
                continue

            # Extraemos la oracion especifica de contexto para este fenotipo
            phenotype_name = phenotypes_with_context[i][0]
            context_sentence = phenotypes_with_context[i][1]

            candidates_text = ""
            for rank, (sim, code, name) in enumerate(sorted_candidates):
                candidates_text += f"{rank+1}. Código: {code} | Nombre: {name} | (Score vectorial: {sim:.2f})\n"

            prompt = f"""Eres un experto en codificación clínica HPO. 
Contexto clínico original del paciente (oración específica):
"{context_sentence}"

Fenotipo extraído a mapear: "{phenotype_name}"

A continuación, tienes los Top {self.top_k} códigos candidatos pre-seleccionados de la ontología HPO:
{candidates_text}
Tu tarea: Selecciona de la lista anterior el ÚNICO código que mejor represente el fenotipo exacto basándote en el contexto clínico.
Si ninguno de los candidatos es adecuado en absoluto, responde "None".
Responde ÚNICAMENTE con un JSON en este formato estricto: {{"hpo_code": "código_elegido"}}"""

            mapped_phenotypes[i] = self._call_llm_rag(prompt)

        return mapped_phenotypes

    def _call_llm_rag(self, prompt: str) -> str:
        response_text = ""
        try:
            if self.api_provider == "gemini":
                interaction = self.client.interactions.create(model=self.api_model_name, input=prompt)
                response_text = interaction.output_text
            elif self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content

            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match: clean_text = match.group(0)
            parsed_data = json.loads(clean_text)
            return parsed_data.get("hpo_code", "None")
        except Exception as e:
            print(f"Error en RAG LLM. Fallback a 'None'. Error: {e}")
            return "None"

    def _get_codes_batch(self):
        batch_files = glob.glob(os.path.join(self.ontology_file_path, "*.pt"))
        for batch_file in batch_files:
            raw_dict = torch.load(batch_file, map_location=self.pytorch_device)
            yield raw_dict
            del raw_dict
            if self.pytorch_device.type == 'cuda': torch.cuda.empty_cache()

    def _get_encoded_phenotypes_list(self, phenotypes_list: list[str]):
        with torch.no_grad():
            tokenized_symptoms = self.tokenizer(
                phenotypes_list, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.pytorch_device)
            phenotypes_model_output = self.model(**tokenized_symptoms)
            phenotypes_cls_embedding = phenotypes_model_output.last_hidden_state[:, 0, :]
            encoded_phenotypes_list = F.normalize(phenotypes_cls_embedding, p=2, dim=1)
        return encoded_phenotypes_list