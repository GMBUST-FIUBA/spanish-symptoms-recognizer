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

def get_llm_api_models_test_config():
    yield {
        "nombre_prueba": f"Gemini 3.6-flash",
        "kwargs": {
            "ontology": "hpo",
            "text_parser" : "full-text",
            "phenotypes_model_type" : "llm_api",
        }
    }

def compare_models():
    dataset_dir = os.path.join(CURRENT_DIR, "dataset")

    TESTED_CONFIGURATIONS = get_llm_api_models_test_config()

    comparison_results = []

    for config in TESTED_CONFIGURATIONS:
        test_name = config["nombre_prueba"]
        print(f"[{test_name}] Iniciando evaluación...")
        
        try:
            recognizer = PhenotypesRecognizer(**config["kwargs"])

            evaluator = Evaluator(recognizer)
            evaluator.evaluate_directory(dataset_dir, csv_hpo_column="hpo_code")

            global_scores = evaluator._calculate_f1(
                evaluator.global_tp, 
                evaluator.global_fp, 
                evaluator.global_fn
            )

            comparison_results.append({
                "Prueba": test_name,
                "Modelo_Path": os.path.basename(config["kwargs"].get("ner_model_path", "")),
                "Agg_Strategy": config["kwargs"].get("agg_strategy", "N/A"),
                "TP": evaluator.global_tp,
                "FP": evaluator.global_fp,
                "FN": evaluator.global_fn,
                "Precision": global_scores["Precision"],
                "Recall": global_scores["Recall"],
                "F1_Score": global_scores["F1_Score"]
            })

            print(f"[{test_name}] Completado. F1-Score: {global_scores['F1_Score']:.4f}\n")
            
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
        df_comparison = df_comparison.sort_values(by="F1_Score", ascending=False).reset_index(drop=True)
        
    return df_comparison

if __name__ == "__main__":
    df_results = compare_models()
    
    if not df_results.empty:
        print("\n" + "=" * 90)
        print("REPORTE COMPARATIVO DE MODELOS Y ESTRATEGIAS".center(90))
        print("=" * 90)
        print(df_results.to_string())
        print("=" * 90)