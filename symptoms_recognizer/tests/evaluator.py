import os
import glob
import pandas as pd
import re
from typing import Set, Dict, List

from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

class Evaluator:
    def __init__(self, recognizer: PhenotypesRecognizer):
        self.recognizer = recognizer

        self.global_code_tp = 0
        self.global_code_fp = 0
        self.global_code_fn = 0

        self.global_text_tp = 0
        self.global_text_fp = 0
        self.global_text_fn = 0
        
        self.results_per_doc = []

    def _calculate_metrics(self, expected: Set[str], predicted: Set[str]) -> Dict[str, int]:
        tp = len(expected.intersection(predicted))
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        return {"TP": tp, "FP": fp, "FN": fn}

    def _calculate_f1(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4)
        }

    def evaluate_directory(self, data_dir: str, csv_hpo_column: str = "hpo_code", csv_text_column: str = "phen_texts"):
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

        for txt_path in txt_files:
            base_name = os.path.splitext(txt_path)[0]
            csv_path = f"{base_name}.csv"

            if not os.path.exists(csv_path):
                print(f"Advertencia: No se encontró el CSV para {txt_path}. Se omite.")
                continue

            df_expected = pd.read_csv(csv_path)
            expected_hpo = set(df_expected[csv_hpo_column].astype(str).str.strip())

            if csv_text_column in df_expected.columns:
                expected_texts = set(df_expected[csv_text_column].dropna().astype(str).str.lower().str.strip())
            else:
                expected_texts = set()

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            predicted_texts_list = self.recognizer.recognize(text)
            predicted_texts = set()
            for item in predicted_texts_list:
                if isinstance(item, tuple) and item[0]:
                    predicted_texts.add(item[0].lower().strip())
                elif isinstance(item, str) and item:
                    predicted_texts.add(item.lower().strip())

            predicted_hpo_list = self.recognizer.map(predicted_texts_list)
            predicted_hpo = set([code.strip() for code in predicted_hpo_list if code != "None"])

            doc_text_metrics = self._calculate_metrics(expected_texts, predicted_texts)
            doc_text_scores = self._calculate_f1(doc_text_metrics["TP"], doc_text_metrics["FP"], doc_text_metrics["FN"])
            
            text_tp_set = expected_texts.intersection(predicted_texts)
            text_fp_set = predicted_texts - expected_texts
            text_fn_set = expected_texts - predicted_texts

            doc_code_metrics = self._calculate_metrics(expected_hpo, predicted_hpo)
            doc_code_scores = self._calculate_f1(doc_code_metrics["TP"], doc_code_metrics["FP"], doc_code_metrics["FN"])
            
            hpo_tp_set = expected_hpo.intersection(predicted_hpo)
            hpo_fp_set = predicted_hpo - expected_hpo
            hpo_fn_set = expected_hpo - predicted_hpo
            
            self.results_per_doc.append({
                "Document": os.path.basename(txt_path),
                "Input_Text": text.strip(),

                "Code_F1": doc_code_scores["F1_Score"],
                "Text_F1": doc_text_scores["F1_Score"],

                "Text_TP": text_tp_set,
                "Text_FP": text_fp_set,
                "Text_FN": text_fn_set,

                "HPO_TP": hpo_tp_set,
                "HPO_FP": hpo_fp_set,
                "HPO_FN": hpo_fn_set
            })

            self.global_code_tp += doc_code_metrics["TP"]
            self.global_code_fp += doc_code_metrics["FP"]
            self.global_code_fn += doc_code_metrics["FN"]
            
            self.global_text_tp += doc_text_metrics["TP"]
            self.global_text_fp += doc_text_metrics["FP"]
            self.global_text_fn += doc_text_metrics["FN"]

    def export_detailed_reports(self, output_dir: str, test_name: str):
        os.makedirs(output_dir, exist_ok=True)

        safe_test_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', test_name)

        ner_file = os.path.join(output_dir, f"{safe_test_name}_NER.txt")
        map_file = os.path.join(output_dir, f"{safe_test_name}_MAP.txt")

        with open(ner_file, "w", encoding="utf-8") as f_ner, open(map_file, "w", encoding="utf-8") as f_map:
            f_ner.write(f"REPORTE DETALLADO NER (Extracción de Texto) - PRUEBA: {test_name}\n")
            f_ner.write("="*80 + "\n\n")

            f_map.write(f"REPORTE DETALLADO MAPEO (Códigos HPO) - PRUEBA: {test_name}\n")
            f_map.write("="*80 + "\n\n")

            for doc in self.results_per_doc:
                doc_name = doc["Document"]
                text_input = doc["Input_Text"]

                f_ner.write(f"DOCUMENTO: {doc_name}\n")
                f_ner.write(f"--- ENTRADA ---\n{text_input}\n\n")

                f_ner.write(f"TP (Aciertos) [{len(doc['Text_TP'])}]:\n")
                for item in doc["Text_TP"]: f_ner.write(f"  - {item}\n")

                f_ner.write(f"FP (Alucinaciones / Predicciones incorrectas) [{len(doc['Text_FP'])}]:\n")
                for item in doc["Text_FP"]: f_ner.write(f"  - {item}\n")

                f_ner.write(f"\nFN (Omitidos en la extracción) [{len(doc['Text_FN'])}]:\n")
                for item in doc["Text_FN"]: f_ner.write(f"  - {item}\n")

                f_ner.write("\n" + "-"*80 + "\n\n")

                f_map.write(f"DOCUMENTO: {doc_name}\n")
                f_map.write(f"--- ENTRADA ---\n{text_input}\n\n")

                f_map.write(f"TP (Códigos Correctos) [{len(doc['HPO_TP'])}]:\n")
                for item in doc["HPO_TP"]: f_map.write(f"  - {item}\n")

                f_map.write(f"\nFP (Códigos Incorrectos / Alucinados) [{len(doc['HPO_FP'])}]:\n")
                for item in doc["HPO_FP"]: f_map.write(f"  - {item}\n")

                f_map.write(f"\nFN (Códigos Omitidos) [{len(doc['HPO_FN'])}]:\n")
                for item in doc["HPO_FN"]: f_map.write(f"  - {item}\n")

                f_map.write("\n" + "-"*80 + "\n\n")

    def get_report(self) -> pd.DataFrame:
        clean_results = []
        for d in self.results_per_doc:
            clean_results.append({
                "Document": d["Document"],
                "Code_F1": d["Code_F1"],
                "Text_F1": d["Text_F1"]
            })
        return pd.DataFrame(clean_results)

    def print_global_metrics(self):
        global_code_scores = self._calculate_f1(self.global_code_tp, self.global_code_fp, self.global_code_fn)
        global_text_scores = self._calculate_f1(self.global_text_tp, self.global_text_fp, self.global_text_fn)

        print("\n" + "="*50)
        print("MÉTRICAS GLOBALES (Micro-Averaged)")
        print("="*50)
        print("--- 1. EXTRACCIÓN DE TEXTO (NER / LLM) ---")
        print(f"TP: {self.global_text_tp} | FP: {self.global_text_fp} | FN: {self.global_text_fn}")
        print(f"Precision : {global_text_scores['Precision']:.4f}")
        print(f"Recall    : {global_text_scores['Recall']:.4f}")
        print(f"F1 Score  : {global_text_scores['F1_Score']:.4f}")

        print("\n--- 2. MAPEO DE CÓDIGOS (HPO) ---")
        print(f"TP: {self.global_code_tp} | FP: {self.global_code_fp} | FN: {self.global_code_fn}")
        print(f"Precision : {global_code_scores['Precision']:.4f}")
        print(f"Recall    : {global_code_scores['Recall']:.4f}")
        print(f"F1 Score  : {global_code_scores['F1_Score']:.4f}")
        print("="*50)