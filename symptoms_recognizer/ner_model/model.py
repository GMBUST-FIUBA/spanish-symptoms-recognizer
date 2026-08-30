from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, GenerationConfig
from transformers import pipeline

import json
import os
import re
import torch

# Local model path
CURRENT_DIR = Path(__file__).parent.resolve()
LOCAL_MODEL_RELATIVE_PATH = "model/base-nat-data"
LOCAL_LLM_RELATIVE_PATH = "model/Qwen2.5-0.5B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)
DEFAULT_LOCAL_LLM_PATH = os.path.join(CURRENT_DIR, LOCAL_LLM_RELATIVE_PATH)

class PhenotypesDetector:
    def __init__(self, model_path=None,
                 tokenizer_path=None,
                 allowed_entity_groups=None,
                 agg_strategy="simple",
                 phenotypes_model_type="ner"):

        self.phenotypes_model_type = phenotypes_model_type

        if phenotypes_model_type == "ner":
            model_path = model_path or DEFAULT_LOCAL_MODEL_PATH
            tokenizer_path = tokenizer_path or model_path
            self.allowed_entity_groups = set(allowed_entity_groups) if allowed_entity_groups else None
    
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy=agg_strategy,
            )

        elif phenotypes_model_type == "llm":
            model_path = model_path or DEFAULT_LOCAL_LLM_PATH
            tokenizer_path = tokenizer_path or model_path

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.ner_pipeline = pipeline(
                "text-generation",
                model=model_path,
                tokenizer=self.tokenizer,
                device_map="auto",
                dtype=torch.bfloat16,
            )
            self.base_prompt = """Eres un asistente médico experto en extraer signos y síntomas clínicos. Tu tarea es extraer los fenotipos positivos del paciente actual a partir del texto y devolverlos estrictamente en formato JSON.

REGLAS:
1. Extrae SOLO los síntomas o signos que el paciente SÍ tiene.
2. IGNORA los síntomas negados (ejemplo: "sin fiebre", "niega dolor").
3. IGNORA los antecedentes de familiares (ejemplo: "madre con asma").
4. Responde ÚNICAMENTE con un objeto JSON, sin texto adicional ni explicaciones.

EJEMPLO DE ENTRADA:
"Paciente presenta cefalea severa y fotofobia. Sin náuseas. Padre con hipertensión."

EJEMPLO DE SALIDA:
{"fenotipos": ["cefalea severa", "fotofobia"]}"""

        else:
            raise Exception("Non-existent model type")

    def detect_phenotypes(self, sentence: str) -> list[str]:
        phenotypes_list = []

        if self.phenotypes_model_type == "ner":
            tokens = self.tokenizer.encode(
                sentence, 
                truncation=True, 
                max_length=512
            )
            truncated_sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)

            sentence_results = self.ner_pipeline(truncated_sentence)

            for res in sentence_results:
                entity_group = res.get("entity_group")
                if not self.allowed_entity_groups or entity_group in self.allowed_entity_groups:
                    phenotypes_list.append(res["word"].strip())

        else:
            if not sentence or not sentence.strip():
                return []

            input_tokens = self.tokenizer.encode(sentence, truncation=True, max_length=2048)
            safe_sentence = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            messages = [
                {
                    "role": "system", 
                    "content": self.base_prompt
                },
                {
                    "role": "user", 
                    "content": f"Texto de entrada:\n{safe_sentence}"
                }
            ]

            try:
                outputs = self.ner_pipeline(
                    messages,
                    max_new_tokens=1536,
                    max_length=None,
                    do_sample=False,
                    return_full_text=False
                )
            except Exception as e:
                print(f"Error interno del pipeline LLM en este chunk: {e}")
                return []

            if not outputs or not isinstance(outputs, list) or len(outputs) == 0:
                return []

            response_text = outputs[0].get("generated_text", "")

            try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                
                match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                if match:
                    clean_text = match.group(0)

                parsed_data = json.loads(clean_text)

                raw_phenotypes = parsed_data.get("fenotipos", [])
                phenotypes_list = list(set(raw_phenotypes))

            except json.JSONDecodeError:
                print(f"Fallo al parsear JSON. Salida cruda del modelo:\n{response_text}")
                phenotypes_list = []

        return phenotypes_list