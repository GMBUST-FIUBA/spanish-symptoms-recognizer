import os
import glob
import pandas as pd
from typing import Set, Dict, List

from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

class Evaluator:
    def __init__(self, recognizer: PhenotypesRecognizer):
        self.recognizer = recognizer

        self.global_tp = 0
        self.global_fp = 0
        self.global_fn = 0
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

    def evaluate_directory(self, data_dir: str, csv_hpo_column: str = "hpo_code"):
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        for txt_path in txt_files:
            base_name = os.path.splitext(txt_path)[0]
            csv_path = f"{base_name}.csv"
            
            if not os.path.exists(csv_path):
                print(f"Advertencia: No se encontró el CSV para {txt_path}. Se omite.")
                continue
                
            df_expected = pd.read_csv(csv_path)
            expected_hpo = set(df_expected[csv_hpo_column].astype(str).str.strip())
            
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            predicted_hpo_list = self.recognizer.scan(text, only_results=True)
            predicted_hpo = set([code.strip() for code in predicted_hpo_list if code != "None"])
            
            doc_metrics = self._calculate_metrics(expected_hpo, predicted_hpo)
            doc_scores = self._calculate_f1(doc_metrics["TP"], doc_metrics["FP"], doc_metrics["FN"])
            
            self.results_per_doc.append({
                "Document": os.path.basename(txt_path),
                **doc_metrics,
                **doc_scores,
                "Missed (FN)": expected_hpo - predicted_hpo,
                "Hallucinated (FP)": predicted_hpo - expected_hpo
            })

            self.global_tp += doc_metrics["TP"]
            self.global_fp += doc_metrics["FP"]
            self.global_fn += doc_metrics["FN"]

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.results_per_doc)

    def print_global_metrics(self):
        global_scores = self._calculate_f1(self.global_tp, self.global_fp, self.global_fn)
        
        print("\n" + "="*40)
        print("MÉTRICAS GLOBALES (Micro-Averaged)")
        print("="*40)
        print(f"Total True Positives (TP) : {self.global_tp}")
        print(f"Total False Positives (FP): {self.global_fp}")
        print(f"Total False Negatives (FN): {self.global_fn}")
        print("-" * 40)
        print(f"Precision : {global_scores['Precision']:.4f}")
        print(f"Recall    : {global_scores['Recall']:.4f}")
        print(f"F1 Score  : {global_scores['F1_Score']:.4f}")
        print("="*40)

if __name__ == "__main__":
    recognizer = PhenotypesRecognizer(ontology="hpo")
    evaluator = Evaluator(recognizer)
    evaluator.evaluate_directory("./symptoms_recognizer/tests/dataset", csv_hpo_column="hpo_code")
    evaluator.print_global_metrics()

    df_reporte = evaluator.get_report()
    print(df_reporte.head())