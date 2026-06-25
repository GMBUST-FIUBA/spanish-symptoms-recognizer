import time
import csv
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModel

# Embeddings generator
## Local path
LOCAL_TOKENIZER_PATH = "semantic_model"
LOCAL_MODEL_PATH = "semantic_model"

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
    # Get GPU to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)

    # Open data file
    hpo_codes_tokens = {}
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

                    # Normalize text
                    clean_text = row[ORIGINAL_TSV_FILE_TRANSALTION_COLUMN].strip()

                    # Tokenize translation using GPU if available
                    tokenized_translation = tokenizer(
                        clean_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)

                    # Embbed tokenized input
                    model_output = model(**tokenized_translation)
                    
                    # Get CLS token
                    cls_embedding = model_output.last_hidden_state[0, 0, :]
                    
                    # Normalize
                    normalized_embedding = F.normalize(cls_embedding.unsqueeze(0), p=2, dim=1).squeeze()
                    
                    # Convert to numpy and take back to CPU to use it
                    sentence_embedding = normalized_embedding.cpu().numpy()

                    # Add row
                    hpo_code = row[ORIGINAL_TSV_FILE_HPO_CODE]
                    hpo_codes_tokens[hpo_code] = sentence_embedding

                print(f"Duración de creación usando torch para guardar: {time.time() - start_time}")

            torch.save(hpo_codes_tokens, output_file)

if __name__ == "__main__":
    generate_tokens_file()