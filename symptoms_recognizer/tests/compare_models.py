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

warnings.filterwarnings("ignore", message="Tokenizer does not support real words")

def compare_models():
    models_dir = os.path.join(PROJECT_ROOT, "symptoms_recognizer", "ner_model", "model")
    dataset_dir = os.path.join(CURRENT_DIR, "dataset")
    
    TESTED_CONFIGURATIONS = [
        {
            "nombre_prueba": "base-nat-data (agg: simple)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "base-nat-data"),
                "ner_tokenizer_path": os.path.join(models_dir, "base-nat-data"),
                "ontology": "hpo",
                "agg_strategy": "simple"
            }
        },
        {
            "nombre_prueba": "base-nat-data (agg: first)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "base-nat-data"),
                "ner_tokenizer_path": os.path.join(models_dir, "base-nat-data"),
                "ontology": "hpo",
                "agg_strategy": "first"
            }
        },
        {
            "nombre_prueba": "HUMADEX (agg: simple)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "HUMADEX"),
                "ner_tokenizer_path": os.path.join(models_dir, "HUMADEX"),
                "ontology": "hpo",
                "agg_strategy": "simple",
                "allowed_entity_groups" : ["PROBLEM"],
            }
        },
        {
            "nombre_prueba": "HUMADEX (agg: first)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "HUMADEX"),
                "ner_tokenizer_path": os.path.join(models_dir, "HUMADEX"),
                "ontology": "hpo",
                "agg_strategy": "first",
                "allowed_entity_groups" : ["PROBLEM"],
            }
        },
        {
            "nombre_prueba": "HUMADEX (agg: average)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "HUMADEX"),
                "ner_tokenizer_path": os.path.join(models_dir, "HUMADEX"),
                "ontology": "hpo",
                "agg_strategy": "average",
                "allowed_entity_groups" : ["PROBLEM"],
            }
        },
        {
            "nombre_prueba": "roberta-es-clinical-trials (agg: simple)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "roberta-es-clinical-trials-umls-7sgs-ner"),
                "ner_tokenizer_path": os.path.join(models_dir, "roberta-es-clinical-trials-umls-7sgs-ner"),
                "ontology": "hpo",
                "agg_strategy": "simple",
                "allowed_entity_groups" : ["DISO"],
            }
        },
        {
            "nombre_prueba": "roberta-es-clinical-trials (agg: first)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "roberta-es-clinical-trials-umls-7sgs-ner"),
                "ner_tokenizer_path": os.path.join(models_dir, "roberta-es-clinical-trials-umls-7sgs-ner"),
                "ontology": "hpo",
                "agg_strategy": "first",
                "allowed_entity_groups" : ["DISO"],
            }
        },
        {
            "nombre_prueba": "roberta-es-clinical-trials (agg: average)",
            "kwargs": {
                "ner_model_path": os.path.join(models_dir, "roberta-es-clinical-trials-umls-7sgs-ner"),
                "ner_tokenizer_path": os.path.join(models_dir, "roberta-es-clinical-trials-umls-7sgs-ner"),
                "ontology": "hpo",
                "agg_strategy": "average",
                "allowed_entity_groups" : ["DISO"],
            }
        },
    ]

    print(f"Se encontraron {len(TESTED_CONFIGURATIONS)} configuraciones para evaluar.\n")
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