import json
import os
import spacy

from spacy.training import offsets_to_biluo_tags

def get_prefixes_of_brat_files(input_path):
    # Get data input folder
    data_input_folder = input_path

    # Lazy iteration
    for folder_entry in os.listdir(data_input_folder):
        # Check if file is txt
        if folder_entry.endswith(".txt"):
            yield folder_entry.removesuffix(".txt")

def merge_overlapping_entities(entities):
    if not entities:
        return []

    entities = sorted(entities, key=lambda x: x[0])
    
    merged = []
    curr_start, curr_end, curr_label = entities[0]

    for next_start, next_end, next_label in entities[1:]:
        if next_start < curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end, curr_label))
            curr_start, curr_end, curr_label = next_start, next_end, next_label

    merged.append((curr_start, curr_end, curr_label))
    return merged

def get_entities_from_ann_file(ann_file_path):
    entities = []
    if not os.path.exists(ann_file_path):
        return entities
    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                label_info = parts[1].split(' ')
                label = label_info[0]
                start = int(label_info[1])
                end = int(label_info[-1])
                entities.append((start, end, label))
    return entities


LABEL_BIO_MAP_FOR_MODEL = {
    "O" : "O",
    "B" : "B-SINTOMA",
    "I" : "I-SINTOMA"
}

def convert_brat_to_bio(input_path, data_output_file):
    # List data and annotations names
    files_prefixes_generator = get_prefixes_of_brat_files(input_path)

    # Get data input folder
    data_input_folder = input_path

    # Get tokenizer
    tokenizer = spacy.load("es_core_news_sm")

    # Iterate over file prefixes
    for prefix in files_prefixes_generator:
        # Get files
        ann_file_path = os.path.join(data_input_folder, prefix + ".ann")
        txt_file_path = os.path.join(data_input_folder, prefix + ".txt")

        # Tokenize text
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            whole_text = f.read()

        # Get entities from annotation file
        entities = get_entities_from_ann_file(ann_file_path)

        # Get tokenized text
        tokenized_text = tokenizer(whole_text)

        snapped_entities = []
        for start, end, label in entities:
            span = tokenized_text.char_span(start, end, label=label, alignment_mode="expand")
            if span is not None:
                snapped_entities.append((span.start_char, span.end_char, label))
            else:
                print(f"DEBUG: No se pudo alinear el span ({start}, {end}) en {prefix}")

        # Merge entities
        merged_entities = merge_overlapping_entities(snapped_entities)

        # Get BILOU tags using spacy
        biluo_tags = offsets_to_biluo_tags(tokenized_text, merged_entities)

        # For each sentence
        for sentence in tokenized_text.sents:
            sentence_tokens = []
            sentence_tags = []

            # For every token in each sentence
            for token in sentence:
                tag = biluo_tags[token.i]
                
                # Convert from BILOU format to model accepted format
                if tag == "O" or tag == "-":
                    tag_id = LABEL_BIO_MAP_FOR_MODEL["O"]
                elif tag.startswith("B-") or tag.startswith("U-"):
                    tag_id = LABEL_BIO_MAP_FOR_MODEL["B"]
                elif tag.startswith("I-") or tag.startswith("L-"):
                    tag_id = LABEL_BIO_MAP_FOR_MODEL["I"]
                else:
                    tag_id = LABEL_BIO_MAP_FOR_MODEL["O"]

                sentence_tokens.append(token.text)
                sentence_tags.append(tag_id)

            if len(sentence_tokens) > 0:
                new_entry_dict = {
                    "tokens": sentence_tokens,
                    "ner_tags": sentence_tags
                }

                json_record = json.dumps(new_entry_dict, ensure_ascii=False)
                data_output_file.write(json_record + '\n')