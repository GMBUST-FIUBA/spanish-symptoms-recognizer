import json
import os
import random
import spacy

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

            sentence_tokens = []
            sentence_tags = []

            # Iterate over words
            for token in sent:
                if not token.text.strip():
                    continue
                
                sentence_tokens.append(token.text)
                
                # Get start and end
                token_txt_start = token.idx
                token_txt_end = token.idx + len(token.text)
                
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


            # Prepare data for storage
            if len(sentence_tokens) > 0:
                record = {"tokens": sentence_tokens, "ner_tags": sentence_tags}

                # Select output
                if secondary_data_output_file is None or random.random() > data_split:
                    primary_data_output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    secondary_data_output_file.write(json.dumps(record, ensure_ascii=False) + '\n')