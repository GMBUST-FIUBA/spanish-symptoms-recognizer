import os
import sys
import gc
import pandas as pd
import warnings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, PROJECT_ROOT)

from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer
from symptoms_recognizer.tests.evaluator import Evaluator 
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Tokenizer does not support real words")

PROMPT_NER_GENERAL = """Eres un anotador clínico experto especializado en la Ontología de Fenotipos Humanos (HPO). Tu tarea es extraer signos, síntomas y anormalidades fenotípicas de la historia clínica.

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

PROMPT_NER_SPECIFIC = """Eres un anotador clínico experto especializado en la Ontología de Fenotipos Humanos (HPO). Tu tarea es extraer signos, síntomas y anormalidades fenotípicas de la historia clínica.

REGLAS ESTRICTAS DE EXTRACCIÓN:
1. EXTRACCIÓN ESPECÍFICA (GRANULARIDAD): Extrae el síntoma manteniendo sus modificadores clínicos clave (tipo, anatomía, lateralidad, severidad). Usa las palabras literales del texto siempre que sea posible.
   - BIEN: "disfagia motora", "poliartritis simétrica", "dolor abdominal severo", "eritema malar clásico".
2. ELIMINA TIEMPOS Y EXPLICACIONES: El fenotipo NO debe contener descripciones de duración, frases conectoras ni explicaciones.
   - MAL: "rigidez matinal de más de una hora" -> BIEN: "rigidez matinal"
   - MAL: "pérdida de cabello difusa que el especialista diagnostica como alopecia" -> BIEN: "alopecia no cicatricial" o "pérdida de cabello difusa"
3. ENFERMEDADES GLOBALES vs. FENOTIPOS: Ignora diagnósticos de enfermedades sistémicas (ej. "Lupus", "Espondilitis"). SÍ extrae manifestaciones específicas de órganos (ej. "pericarditis", "nefritis lúpica", "sacroiliítis bilateral").
4. LABORATORIOS: Ignora anticuerpos (ej. "ANA positivos") y marcadores inflamatorios ("PCR elevada"). SÍ extrae anormalidades fisiológicas base (ej. "proteinuria", "anemia megaloblástica").
5. IGNORA síntomas negados ("sin fiebre") y antecedentes familiares.
6. Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional.

EJEMPLO DE ENTRADA:
"Paciente con lupus. Presenta poliartritis simétrica severa desde hace 2 meses y nefritis. Laboratorio: ANA positivo y proteinuria de 2g. Sin fiebre."

EJEMPLO DE SALIDA:
{
  "fenotipos": [
    {"fenotipo": "poliartritis simétrica severa", "contexto": "Presenta poliartritis simétrica severa desde hace 2 meses y nefritis."},
    {"fenotipo": "nefritis", "contexto": "Presenta poliartritis simétrica severa desde hace 2 meses y nefritis."},
    {"fenotipo": "proteinuria", "contexto": "Laboratorio: ANA positivo y proteinuria de 2g."}
  ]
}"""

NER_PROMPTS_AVAILABLE = {
    "gral-ner-prompt" : PROMPT_NER_GENERAL,
    "specific-ner-prompt" : PROMPT_NER_SPECIFIC,
}

def get_ner_models_test_config():
    models_dir = os.path.join(PROJECT_ROOT, "symptoms_recognizer", "ner_model", "model")

    MODEL_NAMES = ["base-nat-data", "HUMADEX", "roberta-es-clinical-trials-umls-7sgs-ner"]
    AGG_STRATEGIES = ["simple", "first", "average"]
    PARSING_STYLES = ["sentences", "chunks-sentences", "sections-sentences"]

    for model_name in MODEL_NAMES:
        for agg_strat in AGG_STRATEGIES:
            for parsing_style in PARSING_STYLES:
                new_config = {
                    "nombre_prueba": f"{model_name} (agg: {agg_strat}, parsing: {parsing_style})",
                    "kwargs": {
                        "ner_model_path": os.path.join(models_dir, model_name),
                        "ner_tokenizer_path": os.path.join(models_dir, model_name),
                        "ontology": "hpo",
                        "agg_strategy": agg_strat,
                        "text_parser" : parsing_style,
                    }
                }

                yield new_config

def get_llm_models_test_config():
    models_dir = os.path.join(PROJECT_ROOT, "symptoms_recognizer", "ner_model", "model")

    MODEL_NAMES = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"]
    PARSING_STYLES = ["full-text", "chunks-sentences", "sections-sentences"]

    for model_name in MODEL_NAMES:
        for parsing_style in PARSING_STYLES:
            new_config = {
                "nombre_prueba": f"{model_name} (parsing: {parsing_style})",
                "kwargs": {
                    "ner_model_path": os.path.join(models_dir, model_name),
                    "ner_tokenizer_path": os.path.join(models_dir, model_name),
                    "ontology": "hpo",
                    "text_parser" : parsing_style,
                    "phenotypes_model_type" : "llm",
                }
            }

            yield new_config

def get_gemini_api_models_test_config():
    MODEL_NAMES = [
        "gemini-3.5-flash",
    ]

    for model_name in MODEL_NAMES:
        yield {
            "nombre_prueba": f"Gemini - {model_name}",
            "kwargs": {
                "ontology": "hpo",
                "text_parser": "full-text",
                "phenotypes_model_type": "api",
                "ner_api_provider": "gemini",
                "ner_api_model_name": model_name,
            }
        }

def get_openai_api_models_test_config():
    MODEL_NAMES = [
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-4o-mini",
        "gpt-4.1-mini"
    ]

    for model_name in MODEL_NAMES:
        yield {
            "nombre_prueba": f"ChatGPT - {model_name}",
            "kwargs": {
                "ontology": "hpo",
                "text_parser" : "full-text",
                "phenotypes_model_type" : "api", 
                "ner_api_provider": "openai",
                "ner_api_model_name" : model_name
            }
        }

def get_claude_api_models_test_config():
    MODEL_NAMES = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]

    for model_name in MODEL_NAMES:
        yield {
            "nombre_prueba": f"Claude - {model_name}",
            "kwargs": {
                "ontology": "hpo",
                "text_parser" : "full-text",
                "phenotypes_model_type" : "api", 
                "ner_api_provider": "anthropic",
                "ner_api_model_name" : model_name
            }
        }

def get_ner_api_models_config():
    yield from get_gemini_api_models_test_config()

    yield from get_openai_api_models_test_config()

    yield from get_claude_api_models_test_config()

def get_gemini_test_api_models_ner_and_map_config():
    NER_MODEL_NAMES = [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    ]

    MAP_MODEL_NAMES = [
        "gemini-3.5-flash",
    ]

    for ner_model_name in NER_MODEL_NAMES:
        for map_model_name in MAP_MODEL_NAMES:
            yield {
                "nombre_prueba": f"Gemini - NER: {ner_model_name} - MAP: {map_model_name}",
                "kwargs": {
                    "ontology": "hpo",
                    "text_parser": "full-text",
                    "phenotypes_model_type": "api",

                    "ner_api_provider": "gemini",
                    "ner_api_model_name": ner_model_name,

                    "map_api_provider": "gemini",
                    "map_api_model_name": map_model_name,
                }
            }

def get_gemini_map_api_models_test_config():
    MAP_MODEL_NAMES = [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    ]
    for ner_prompt in NER_PROMPTS_AVAILABLE:
        for map_model_name in MAP_MODEL_NAMES:
            yield {
                "nombre_prueba": f"NER: gpt-4o-mini ({ner_prompt}) | MAP: Gemini ({map_model_name})",
                "kwargs": {
                    "ontology": "hpo",
                    "text_parser": "full-text",
                    "phenotypes_model_type": "api",

                    "ner_api_provider": "openai",
                    "ner_api_model_name": "gpt-4o-mini",

                    "map_api_provider": "gemini",
                    "map_api_model_name": map_model_name,

                    "ner_prompt": NER_PROMPTS_AVAILABLE[ner_prompt]
                }
            }

def get_openai_map_api_models_test_config():
    MAP_MODEL_NAMES = [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4-nano",
        "gpt-5.4-mini",
        "gpt-5-2025-08-07",
        "gpt-5-mini",
        "gpt-4o-mini",
    ]

    for ner_prompt in NER_PROMPTS_AVAILABLE:
        for map_model_name in MAP_MODEL_NAMES:
            yield {
                "nombre_prueba": f"NER: gpt-4o-mini ({ner_prompt}) | MAP: OpenAI ({map_model_name})",
                "kwargs": {
                    "ontology": "hpo",
                    "text_parser": "full-text",
                    "phenotypes_model_type": "api",

                    "ner_api_provider": "openai",
                    "ner_api_model_name": "gpt-4o-mini",

                    "map_api_provider": "openai",
                    "map_api_model_name": map_model_name,

                    "ner_prompt": NER_PROMPTS_AVAILABLE[ner_prompt]
                }
            }

def get_claude_map_api_models_test_config():
    MAP_MODEL_NAMES = ["claude-sonnet-5", "claude-opus-5"]

    for ner_prompt in NER_PROMPTS_AVAILABLE:
        for map_model_name in MAP_MODEL_NAMES:
            yield {
                "nombre_prueba": f"NER: gpt-4o-mini {ner_prompt} | MAP: Claude ({map_model_name})",
                "kwargs": {
                    "ontology": "hpo",
                    "text_parser": "full-text",
                    "phenotypes_model_type": "api",

                    "ner_api_provider": "openai",
                    "ner_api_model_name": "gpt-4o-mini",

                    "map_api_provider": "anthropic",
                    "map_api_model_name": map_model_name,

                    "ner_prompt": NER_PROMPTS_AVAILABLE[ner_prompt]
                }
            }

def get_map_api_models_config():
    yield from get_gemini_map_api_models_test_config()

    yield from get_openai_map_api_models_test_config()

    yield from get_claude_map_api_models_test_config()

def compare_models():
    dataset_dir = os.path.join(CURRENT_DIR, "dataset")
    reports_dir = os.path.join(CURRENT_DIR, "reports") 
    
    TESTED_CONFIGURATIONS = get_map_api_models_config()

    comparison_results = []

    for config in TESTED_CONFIGURATIONS:
        test_name = config["nombre_prueba"]
        print(f"[{test_name}] Iniciando evaluación...")

        try:
            recognizer = PhenotypesRecognizer(**config["kwargs"])
            evaluator = Evaluator(recognizer)

            evaluator.evaluate_directory(
                dataset_dir, 
                csv_hpo_column="hpo_code",
                csv_text_column="phen_texts"
            )

            evaluator.export_detailed_reports(output_dir=reports_dir, test_name=test_name)

            text_scores = evaluator._calculate_f1(
                evaluator.global_text_tp, 
                evaluator.global_text_fp, 
                evaluator.global_text_fn
            )

            code_scores = evaluator._calculate_f1(
                evaluator.global_code_tp, 
                evaluator.global_code_fp, 
                evaluator.global_code_fn
            )

            comparison_results.append({
                "Prueba": test_name,

                "NER TP": evaluator.global_text_tp,
                "NER FP": evaluator.global_text_fp,
                "NER FN": evaluator.global_text_fn,
                "NER Prec": text_scores["Precision"],
                "NER Rec": text_scores["Recall"],
                "NER F1": text_scores["F1_Score"],

                "Map TP": evaluator.global_code_tp,
                "Map FP": evaluator.global_code_fp,
                "Map FN": evaluator.global_code_fn,
                "Map Prec": code_scores["Precision"],
                "Map Rec": code_scores["Recall"],
                "Map F1": code_scores["F1_Score"]
            })

            print(f"Evaluación terminada. Reportes guardados en: {reports_dir}")

        except Exception as e:
            print(f"Error evaluando la configuración '{test_name}': {str(e)}")

        finally:
            if 'recognizer' in locals(): del recognizer
            if 'evaluator' in locals(): del evaluator
            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_comparison = pd.DataFrame(comparison_results)
    if not df_comparison.empty:
        df_comparison = df_comparison.sort_values(by="Map F1", ascending=False).reset_index(drop=True)

    return df_comparison

if __name__ == "__main__":
    df_results = compare_models()

    if not df_results.empty:
        print("\n" + "=" * 120)
        print("REPORTE COMPARATIVO DE MODELOS (TEXTO vs HPO)".center(120))
        print("=" * 120)
        print(df_results.to_string())
        print("=" * 120)