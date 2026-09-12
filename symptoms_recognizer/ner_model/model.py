from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from symptoms_recognizer.api_clients import build_api_client
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
                 phenotypes_model_type="ner",
                 api_provider=None,
                 api_model_name=None,
                 prompt=None):

        self.phenotypes_model_type = phenotypes_model_type
        self.api_provider = api_provider
        self.api_model_name = api_model_name

        if prompt:
            self.base_prompt = prompt
        else:
            self.base_prompt = """Eres un anotador clínico experto especializado en la Ontología de Fenotipos Humanos (HPO). Tu tarea es extraer signos, síntomas y anormalidades fenotípicas de la historia clínica.

REGLAS ESTRICTAS:
1. SOLO FENOTIPOS Y ANORMALIDADES: Extrae manifestaciones clínicas, signos físicos y síntomas reportados u observados.
2. ENFERMEDADES GLOBALES vs. FENOTIPOS ESPECÍFICOS: Ignora los diagnósticos de enfermedades sistémicas o síndromes globales (ej. "Lupus eritematoso sistémico", "Esclerosis sistémica"). Sin embargo, SÍ DEBES extraer anormalidades estructurales o inflamaciones específicas de órganos (ej. "pericarditis", "nefritis", "poliartritis").
3. MANEJO DE LABORATORIOS: No extraigas nombres de anticuerpos (ej. "ANA positivos", "anti-Scl70") ni valores numéricos crudos. SÍ puedes extraer alteraciones de laboratorio normalizadas que representen un fenotipo (ej. "proteinuria", "leucopenia").
4. NORMALIZACIÓN EXTREMA: El valor de "fenotipo" debe ser el concepto médico estandarizado más corto posible (idealmente 1 a 3 palabras). 
   - MAL: "rigidez matinal de más de una hora" -> BIEN: "rigidez matinal"
   - MAL: "proteinuria patológica (1.8 g/24h)" -> BIEN: "proteinuria"
   - MAL: "eritema fijo, plano, de bordes netos" -> BIEN: "eritema malar"
5. IGNORA todo síntoma negado ("sin fiebre") y antecedentes familiares.
6. Devuelve ÚNICAMENTE un objeto JSON válido, sin texto antes ni después.

EJEMPLO DE ENTRADA:
"Paciente con lupus. Presenta poliartritis simétrica y nefritis severa. Laboratorio: ANA positivo y proteinuria de 2g. Sin fiebre."

EJEMPLO DE SALIDA:
{
  "fenotipos": [
    {"fenotipo": "poliartritis", "contexto": "Presenta poliartritis simétrica y nefritis severa."},
    {"fenotipo": "nefritis", "contexto": "Presenta poliartritis simétrica y nefritis severa."},
    {"fenotipo": "proteinuria", "contexto": "Laboratorio: ANA positivo y proteinuria de 2g."}
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

            self._detect_fn = self._detect_ner

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

            self._detect_fn = self._detect_local_llm

        elif phenotypes_model_type == "api":
            if not api_model_name:
                raise ValueError("api_model_name es obligatorio cuando phenotypes_model_type='api'")

            self.client = build_api_client(self.api_provider)

            api_call_fns = {
                "gemini": self._call_gemini_ner,
                "openai": self._call_openai_ner,
                "anthropic": self._call_anthropic_ner,
            }
            self._api_call_fn = api_call_fns[self.api_provider]
            self._detect_fn = self._detect_api

        else:
            raise Exception("Non-existent model type")

    def detect_phenotypes(self, text_chunk: str) -> list[tuple[str, str]]:
        return self._detect_fn(text_chunk)

    def _detect_ner(self, text_chunk: str) -> list[tuple[str, str]]:
        tokens = self.tokenizer.encode(
            text_chunk,
            truncation=True,
            max_length=512
        )
        truncated_sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)
        sentence_results = self.ner_pipeline(truncated_sentence)

        phenotypes_list = []
        for res in sentence_results:
            entity_group = res.get("entity_group")
            if not self.allowed_entity_groups or entity_group in self.allowed_entity_groups:
                phenotypes_list.append((res["word"].strip(), text_chunk))

        return phenotypes_list

    def _detect_local_llm(self, text_chunk: str) -> list[tuple[str, str]]:
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
                return self._parse_llm_json_output(response_text, text_chunk)

        except Exception as e:
            print(f"Error interno del pipeline LLM local en este chunk: {e}")

        return []

    def _detect_api(self, text_chunk: str) -> list[tuple[str, str]]:
        if not text_chunk or not text_chunk.strip():
            return []

        response_text = ""
        try:
            response_text = self._api_call_fn(text_chunk)
            return self._parse_llm_json_output(response_text, text_chunk)

        except Exception as e:
            print(f"Error interno del pipeline API de {self.api_provider}: {e}")

        return []

    def _call_gemini_ner(self, text_chunk: str) -> str:
        full_prompt = f"{self.base_prompt}\n\nTexto de entrada:\n{text_chunk}"
        interaction = self.client.interactions.create(
            model=self.api_model_name,
            input=full_prompt
        )
        return interaction.output_text

    def _call_openai_ner(self, text_chunk: str) -> str:
        response = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=[
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": f"Texto de entrada:\n{text_chunk}"}
            ]
        )
        return response.choices[0].message.content

    def _call_anthropic_ner(self, text_chunk: str) -> str:
        response = self.client.messages.create(
            model=self.api_model_name,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": self.base_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": f"Texto de entrada:\n{text_chunk}"}
            ]
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _parse_llm_json_output(self, response_text: str, fallback_context: str) -> list[tuple[str, str]]:
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
                    extracted_list.append((item.strip(), fallback_context))
                    
        except json.JSONDecodeError:
            print(f"Fallo al parsear JSON. Salida cruda del modelo:\n{response_text}")

        return list(set(extracted_list))