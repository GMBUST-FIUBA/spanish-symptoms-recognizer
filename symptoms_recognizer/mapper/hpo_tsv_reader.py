import time
import csv
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

# Embeddings generator
## Local path
LOCAL_TOKENIZER_PATH = "../base_model/embedding/roberta-base-biomedical-clinical-es"
LOCAL_MODEL_PATH = "../output/model"

## Files for token generation
ORIGINAL_TSV_FILE = "hpo/hp-es.babelon.tsv"
OUTPUT_FILE = "hpo/hpo-tokens.pt"

## Columns
ORIGINAL_TSV_FILE_TRANSALTION_COLUMN = "translation_value"
ORIGINAL_TSV_FILE_HPO_CODE = "subject_id"

## New columns
NEW_TOKENS_FILE_HPO_CODE_COLUMN = "hpo_code"
NEW_TOKENS_FILE_TRANSLATION_COLUMN = "tokens"

field_names=[NEW_TOKENS_FILE_HPO_CODE_COLUMN, NEW_TOKENS_FILE_TRANSLATION_COLUMN]

def generate_tokens_file():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)

    # Open data file
    with open(ORIGINAL_TSV_FILE, "r", encoding="utf-8") as original_data_file:
        # Get reader
        input_tsv_reader = csv.DictReader(original_data_file, delimiter='\t')

        # Open output file
        with open(OUTPUT_FILE, "wb") as output_file:
            # Read all rows
            with torch.no_grad():

                print("Creación de tokens usando torch para guardar")
                start_time = time.time()
                for row in input_tsv_reader:
                    # Get translation embedding
                    tokenized_translation = tokenizer(row[ORIGINAL_TSV_FILE_TRANSALTION_COLUMN], return_tensors="pt", padding=True, truncation=True, max_length=512)
                    model_output = model(**tokenized_translation)
                    sentence_embedding = model_output.last_hidden_state[0, 0, :].numpy()

                    # Store row
                    torch.save({row[ORIGINAL_TSV_FILE_HPO_CODE]: sentence_embedding}, output_file)
                print(f"Duración de creación usando torch para guardar: {time.time() - start_time}")

if __name__ == "__main__":
    generate_tokens_file()