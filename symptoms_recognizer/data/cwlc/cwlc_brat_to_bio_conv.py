import json
import os
import random
import spacy

from transformers import AutoTokenizer

tokenizer_path = "/home/gonzalo/Escritorio/Facultad/Trabajo profesional/spanish-symptoms-recognizer/base_model/bsc-bio-ehr-es"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

nlp = spacy.load("es_core_news_sm", disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe("sentencizer")

# Lazy iterator of prefixes
def get_prefixes_of_brat_files(input_path):
    # Get data input folder
    data_input_folder = input_path

    # Lazy iteration
    for folder_entry in os.listdir(data_input_folder):
        # Check if file is txt
        if folder_entry.endswith(".txt"):
            yield folder_entry.removesuffix(".txt")

# Open .ann file and store entities
def get_entities_from_ann_file(ann_file_path):
    entities = []
    if not os.path.exists(ann_file_path): return entities
    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                label_info = parts[1].split(' ')
                if label_info[0] == "Sign_or_Symptom":
                    entities.append((int(label_info[1]), int(label_info[-1])))
    return entities

# Merge entities
def merge_entities(entities):
    if not entities: return []
    entities = sorted(entities, key=lambda x: x[0])
    merged = []
    curr_start, curr_end = entities[0]
    for next_start, next_end in entities[1:]:
        if next_start < curr_end: # Hay solapamiento o anidamiento
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged

# Convert file to BIO format for the model
def convert_brat_to_bio(input_path, primary_data_output_file, secondary_data_output_file=None, data_split=0.3):
    # Get prefixes
    prefixes = get_prefixes_of_brat_files(input_path)

    # Iterate over files prefixes
    for prefix in prefixes:
        txt_file_path = os.path.join(input_path, prefix + ".txt")
        ann_file_path = os.path.join(input_path, prefix + ".ann")

        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        entities = get_entities_from_ann_file(ann_file_path)
        entities = merge_entities(entities)

        # Get document using SpaCy to get the text sentences
        doc = nlp(text)

        # Iterate over document sentences
        for sent in doc.sents:

            # Check if sentences are empty
            if not sent.text.strip():
                continue

            # Encode sentence
            encoded_sentence = tokenizer(
                sent.text,
                max_length=512,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=False
                )
            tokens_offsets = encoded_sentence["offset_mapping"]
            sentence_tokens = [sent.text[s:e] for s, e in tokens_offsets]

            sentence_tags = []

            # Iterate over all tokens
            sent_start_idx = sent.start_char
            for i, (start, end) in enumerate(tokens_offsets):
                if start == end:
                    sentence_tags.append("O") 
                    continue
                
                token_txt_start = start + sent_start_idx
                token_txt_end = end + sent_start_idx
                
                # For every entity
                tag = "O"
                for entity_start, entity_end in entities:
                    # If token overlaps with entity
                    if token_txt_start < entity_end and token_txt_end > entity_start:
                        # If entity starts on token start
                        if token_txt_start == entity_start:
                            tag = "B-SINTOMA"
                        else:
                            tag = "I-SINTOMA"
                        break

                sentence_tags.append(tag)

            # Just to check if there are wrongly annotated beginnings
            for j in range(len(sentence_tags)):
                if sentence_tags[j] == "I-SINTOMA":
                    if j == 0 or sentence_tags[j-1] == "O":
                        sentence_tags[j] = "B-SINTOMA"


            # Take out elements that make noise
            final_tokens = []
            final_tags = []
            
            for token, tag in zip(sentence_tokens, sentence_tags):
                if token.strip():
                    final_tokens.append(token)
                    final_tags.append(tag)

            if len(final_tokens) > 0:
                # Prepare data for storage
                record = {"tokens": final_tokens, "ner_tags": final_tags}

                # Select output
                if secondary_data_output_file is None or random.random() > data_split:
                    primary_data_output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    secondary_data_output_file.write(json.dumps(record, ensure_ascii=False) + '\n')