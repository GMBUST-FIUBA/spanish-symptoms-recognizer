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
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]

    for model_name in MODEL_NAMES:
        yield {
            "nombre_prueba": f"Gemini - {model_name}",
            "kwargs": {
                "ontology": "hpo",
                "text_parser": "full-text",
                "phenotypes_model_type": "api",
                "api_provider": "gemini",
                "api_model_name": model_name,
            }
        }

def get_openai_api_models_test_config():
    MODEL_NAMES = ["gpt-5.4"]

    for model_name in MODEL_NAMES:
        yield {
            "nombre_prueba": f"ChatGPT {model_name}",
            "kwargs": {
                "ontology": "hpo",
                "text_parser" : "full-text",
                "phenotypes_model_type" : "openai_api",
                "api_model_name" : model_name
            }
        }

def compare_models():
    dataset_dir = os.path.join(CURRENT_DIR, "dataset")

    TESTED_CONFIGURATIONS = get_gemini_api_models_test_config()
    
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