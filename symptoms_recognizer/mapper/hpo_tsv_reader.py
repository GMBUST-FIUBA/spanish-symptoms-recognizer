import time
import csv
import torch
import torch.nn.functional as F
import os

from transformers import AutoTokenizer, AutoModel

# Embeddings generator
## Local path
LOCAL_TOKENIZER_PATH = "semantic_model/clinlinker-kb-gp"
LOCAL_MODEL_PATH = "semantic_model/clinlinker-kb-gp"

## Files for token generation
ORIGINAL_TSV_FILE = "hpo/hp-es.babelon.tsv"
# AHORA GUARDAMOS EN UNA CARPETA, NO EN UN SOLO ARCHIVO
OUTPUT_DIR = "hpo/hpo_batches"

## Columns
ORIGINAL_TSV_FILE_TRANSALTION_COLUMN = "translation_value"
ORIGINAL_TSV_FILE_HPO_CODE = "subject_id"

BATCH_SIZE = 500

def generate_tokens_file():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hpo_codes_tokens = {}
    batch_count = 0

    with open(ORIGINAL_TSV_FILE, "r", encoding="utf-8") as original_data_file:
        input_tsv_reader = csv.DictReader(original_data_file, delimiter='\t')

        with torch.no_grad():
            print("Creación de tokens usando torch para guardar en lotes")
            start_time = time.time()
            
            for row in input_tsv_reader:
                clean_text = row[ORIGINAL_TSV_FILE_TRANSALTION_COLUMN].strip()

                tokenized_translation = tokenizer(
                    clean_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                model_output = model(**tokenized_translation)
                cls_embedding = model_output.last_hidden_state[0, 0, :]
                sentence_embedding = F.normalize(cls_embedding.unsqueeze(0), p=2, dim=1).squeeze().cpu()
                hpo_code = row[ORIGINAL_TSV_FILE_HPO_CODE]

                hpo_codes_tokens[hpo_code] = {
                    "vector": sentence_embedding,
                    "name": clean_text
                }

                if len(hpo_codes_tokens) == BATCH_SIZE:
                    output_file_path = os.path.join(OUTPUT_DIR, f"hpo_batch_{batch_count}.pt")
                    torch.save(hpo_codes_tokens, output_file_path)
                    print(f"Guardado lote {batch_count} con 500 códigos.")
                    
                    hpo_codes_tokens = {}
                    batch_count += 1

            if hpo_codes_tokens:
                output_file_path = os.path.join(OUTPUT_DIR, f"hpo_batch_{batch_count}.pt")
                torch.save(hpo_codes_tokens, output_file_path)
                print(f"Guardado lote final {batch_count} con {len(hpo_codes_tokens)} códigos.")

            print(f"Duración total: {time.time() - start_time}")

if __name__ == "__main__":
    generate_tokens_file()