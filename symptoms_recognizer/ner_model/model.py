from google import genai
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import json
import os
import re
import torch
from openai import OpenAI

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
                 phenotypes_model_type="ner",
                 api_provider=None,
                 api_model_name=None):

        self.phenotypes_model_type = phenotypes_model_type
        self.api_provider = api_provider
        self.api_model_name = api_model_name

        # NUEVO PROMPT: Pide explícitamente el fenotipo Y el contexto (la oración)
        self.base_prompt = """Eres un asistente médico experto en extraer signos y síntomas clínicos. Tu tarea es extraer los fenotipos positivos del paciente actual a partir del texto y devolverlos estrictamente en formato JSON.

REGLAS:
1. Extrae SOLO los síntomas o signos que el paciente SÍ tiene.
2. IGNORA los síntomas negados (ejemplo: "sin fiebre", "niega dolor").
3. IGNORA los antecedentes de familiares (ejemplo: "madre con asma").
4. Para cada fenotipo, extrae también la oración o fragmento exacto del texto original donde se menciona (como 'contexto').
5. Responde ÚNICAMENTE con un objeto JSON, sin texto adicional ni explicaciones, siguiendo exactamente la estructura del ejemplo.

EJEMPLO DE ENTRADA:
"Paciente de 45 años. Presenta cefalea severa y fotofobia desde ayer. Sin náuseas. Padre con hipertensión."

EJEMPLO DE SALIDA:
{
  "fenotipos": [
    {"fenotipo": "cefalea severa", "contexto": "Presenta cefalea severa y fotofobia desde ayer."},
    {"fenotipo": "fotofobia", "contexto": "Presenta cefalea severa y fotofobia desde ayer."}
  ]
}"""

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

        elif phenotypes_model_type == "api":
            if not api_model_name:
                raise ValueError("api_model_name es obligatorio cuando phenotypes_model_type='api'")

            if self.api_provider == "gemini":
                self.client = genai.Client()
            elif self.api_provider == "openai":
                self.client = OpenAI()
            else:
                raise Exception(f"Proveedor de API no soportado: {self.api_provider}")

        else:
            raise Exception("Non-existent model type")

    # RENOMBRADO: 'sentence' a 'text_chunk' porque puede ser el texto completo
    def detect_phenotypes(self, text_chunk: str) -> list[tuple[str, str]]:
        phenotypes_list = []

        if self.phenotypes_model_type == "ner":
            tokens = self.tokenizer.encode(
                text_chunk, 
                truncation=True, 
                max_length=512
            )
            truncated_sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)
            sentence_results = self.ner_pipeline(truncated_sentence)

            for res in sentence_results:
                entity_group = res.get("entity_group")
                if not self.allowed_entity_groups or entity_group in self.allowed_entity_groups:
                    # NER tradicional no separa oraciones fácil, así que usamos el chunk como contexto
                    phenotypes_list.append((res["word"].strip(), text_chunk))

        elif self.phenotypes_model_type == "api":
            if not text_chunk or not text_chunk.strip():
                return []

            response_text = ""
            full_prompt = f"{self.base_prompt}\n\nTexto de entrada:\n{text_chunk}"

            try:
                if self.api_provider == "gemini":
                    interaction = self.client.interactions.create(
                        model=self.api_model_name,
                        input=full_prompt
                    )
                    response_text = interaction.output_text
                    
                elif self.api_provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=[
                            {"role": "system", "content": self.base_prompt},
                            {"role": "user", "content": f"Texto de entrada:\n{text_chunk}"}
                        ]
                    )
                    response_text = response.choices[0].message.content

                phenotypes_list = self._parse_llm_json_output(response_text, text_chunk)

            except Exception as e:
                print(f"Error interno del pipeline API de {self.api_provider}: {e}")

        elif self.phenotypes_model_type == "llm":
            if not text_chunk or not text_chunk.strip():
                return []

            input_tokens = self.tokenizer.encode(text_chunk, truncation=True, max_length=2048)
            safe_sentence = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": f"Texto de entrada:\n{safe_sentence}"}
            ]

            try:
                outputs = self.ner_pipeline(
                    messages,
                    max_new_tokens=1536,
                    max_length=None,
                    do_sample=False,
                    return_full_text=False
                )
                
                if outputs and isinstance(outputs, list) and len(outputs) > 0:
                    response_text = outputs[0].get("generated_text", "")
                    phenotypes_list = self._parse_llm_json_output(response_text, text_chunk)

            except Exception as e:
                print(f"Error interno del pipeline LLM local en este chunk: {e}")

        return phenotypes_list

    def _parse_llm_json_output(self, response_text: str, fallback_context: str) -> list[tuple[str, str]]:
        """
        Lógica unificada para extraer la tupla (fenotipo, contexto) del JSON del LLM.
        """
        extracted_list = []
        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                clean_text = match.group(0)

            parsed_data = json.loads(clean_text)
            raw_items = parsed_data.get("fenotipos", [])

            for item in raw_items:
                if isinstance(item, dict):
                    feno = item.get("fenotipo", "").strip()
                    ctx = item.get("contexto", fallback_context).strip()
                    if feno:
                        extracted_list.append((feno, ctx))
                elif isinstance(item, str):
                    # Fallback por si el LLM alucina y devuelve una lista de strings
                    extracted_list.append((item.strip(), fallback_context))
                    
        except json.JSONDecodeError:
            print(f"Fallo al parsear JSON. Salida cruda del modelo:\n{response_text}")
            
        # Deduplicamos usando un set, manteniendo el formato (fenotipo, contexto)
        return list(set(extracted_list))