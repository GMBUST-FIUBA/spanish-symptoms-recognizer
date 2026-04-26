import argparse
import json
import os
import spacy

from spacy.training import offsets_to_biluo_tags

def get_prefixes_of_brat_files(args):
    # Get data input folder
    data_input_folder = args.input_folder

    # Lazy iteration
    for folder_entry in os.listdir(data_input_folder):
        # Check if file is txt
        if folder_entry.endswith(".txt"):
            yield folder_entry.removesuffix(".txt")

def filter_overlapping_entities(entities):
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered = []
    last_end = -1
    
    for start, end, label in entities:
        # Check overlapping. If there is not, append entity.
        # Otherwhise ignore
        if start >= last_end:
            filtered.append((start, end, label))
            last_end = end
            
    return filtered


def get_entities_from_ann_file(ann_file_path):
    entities = []
    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # The entities lines start with T
            if line.startswith('T'):
                parsed_line = line.strip().split('\t')

                # I get the information about the entity in the text
                symptom_info = parsed_line[1].split(' ')

                symptom = symptom_info[0]
                start = int(symptom_info[1])
                end = int(symptom_info[2])

                entities.append((start, end, symptom))

    return entities


LABEL_BIO_MAP_FOR_MODEL = {
    "O" : "O",
    "B" : "B-SINTOMA",
    "I" : "I-SINTOMA"
}

def convert_brat_to_bio(args):
    # Print args:
    print(args.input_folder)
    print(args.output_file)

    # List data and annotations names
    files_prefixes_generator = get_prefixes_of_brat_files(args)

    # Get data input folder
    data_input_folder = args.input_folder

    # Get output file
    data_output_file_path = args.output_file
    with open(data_output_file_path, "w") as data_output_file:

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
            entities = filter_overlapping_entities(entities)

            # Get BILOU tags using spacy
            tokenized_text = tokenizer(whole_text)
            biluo_tags = offsets_to_biluo_tags(tokenized_text, entities)

            # For each sentence
            for sentence in tokenized_text.sents:
                sentence_tokens = []
                sentence_tags = []

                # For every token in each sentence
                for token in sentence:
                    tag = biluo_tags[token.i] # Recuperamos la etiqueta por el índice del token
                    
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

                # Solo agregamos oraciones que no estén vacías
                if len(sentence_tokens) > 0:
                    new_entry_dict = {
                        "tokens": sentence_tokens,
                        "ner_tags": sentence_tags
                    }

                    json_record = json.dumps(new_entry_dict, ensure_ascii=False)
                    data_output_file.write(json_record + '\n')


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Input folder where the annotations and texts are")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument("--max-length", default=512, help="Output file path")
    args = parser.parse_args()

    # Create single file of data
    convert_brat_to_bio(args)