import argparse
import csv
import json
import random
import re
from transformers import AutoTokenizer

tokenizer_path = "/home/gonzalo/Escritorio/Facultad/Trabajo profesional/spanish-symptoms-recognizer/symptoms_recognizer/ner_model/base_model/bsc-bio-ehr-es"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def convert_first_char_to_lower_case(text: str):
    return text.replace(text[0], text[0].lower(), 1)

def get_phenotypes():
    with open("input_texts/hp-es.babelon.tsv", mode="r", encoding="utf-8") as file:
        file_reader = csv.DictReader(file, delimiter="\t")
        next(file_reader) # To avoid the "All" category

        for row in file_reader:
            yield {
                "code" : row["subject_id"],
                "name" : row["translation_value"]
                }

def get_sentences_templates():
    sentences_templates = []

    # Get templates
    with open("input_texts/sentences_templates.txt", mode="r", encoding="utf-8") as file:
        for template in file:
            total_variables = {}
            internal_variables = []

            def change_variable_name(match):
                template_tag = match.group(1)
                total_variables[template_tag] = total_variables.get(template_tag, 0) + 1
                new_elem_name = f"{template_tag}_{total_variables[template_tag]}"
                internal_variables.append(new_elem_name)
                return f"{{{new_elem_name}}}"

            new_template = re.sub(r"<([^>]+)>", change_variable_name, template)

            sentences_templates.append((new_template.strip(), internal_variables))

    return sentences_templates

POSSIBLE_TIMES = [
    "00:00",
    "00:30",
    "01:00",
    "01:30",
    "02:00",
    "02:30",
    "03:00",
    "03:30",
    "04:00",
    "04:30",
    "05:00",
    "05:30",
    "06:00",
    "06:30",
    "07:00",
    "07:30",
    "08:00",
    "08:30",
    "09:00",
    "09:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
    "16:30",
    "17:00",
    "17:30",
    "18:00",
    "18:30",
    "19:00",
    "19:30",
    "20:00",
    "20:30",
    "21:00",
    "21:30",
    "22:00",
    "22:30",
    "23:00",
    "23:30",
]

POSSIBLE_MEDICAL_PLACES = [
    "l hospital",
    "l sanatorio",
    " la clínica",
    " la guardia",
]

POSSIBLE_DAY_TIMES = [
    "mañana",
    "tarde",
    "noche",
    "madrugada",
]

POSSIBLE_COMPANIONS = [
    "un amigo",
    "un conocido",
    "un familiar",
    "el hijo",
    "la hija",
    "la madre",
    "el padre",
    "el abuelo",
    "la abuela",
    "el nieto",
    "la nieta",
]

POSSIBLE_PASSED_TIME = [
    "un día",
    "dos días",
    "2 días",
    "una semana",
    "2 semanas",
    "dos semanas",
    "un mes",
    "dos meses",
    "2 meses",
    "tres meses",
    "3 meses",
]

POSSIBLE_PATIENT_SEX = [
    "hombre",
    "mujer",
]

def generate_medical_history_sentences(phenotype_name, sentences_templates):

    for template_sentence, template_variables in sentences_templates:

        # Iterate over variables of sentence
        sentence_variables_values = {}
        inserted_phenotype = None

        for var in template_variables:
            if "FENOTIPO" in var:
                inserted_phenotype = convert_first_char_to_lower_case(phenotype_name)
                sentence_variables_values[var] = inserted_phenotype
            elif "HORA" in var:
                time = random.choice(POSSIBLE_TIMES)
                sentence_variables_values[var] = time
            elif "LUGAR_MEDICO" in var:
                place = random.choice(POSSIBLE_MEDICAL_PLACES)
                sentence_variables_values[var] = place
            elif "ETAPA_DIA" in var:
                daytime = random.choice(POSSIBLE_DAY_TIMES)
                sentence_variables_values[var] = daytime
            elif "ACOMPANANTE" in var:
                companion = random.choice(POSSIBLE_COMPANIONS)
                if "ACOMPANANTE_INICIO" in var:
                    companion = convert_first_char_to_lower_case(companion)
                sentence_variables_values[var] = companion
            elif "TIEMPO_PASADO" in var:
                passed_time = random.choice(POSSIBLE_PASSED_TIME)
                sentence_variables_values[var] = passed_time
            elif "SEXO_PACIENTE" in var:
                patient_sex = random.choice(POSSIBLE_PATIENT_SEX)
                sentence_variables_values[var] = patient_sex
            else:
                raise Exception("Unknown variable found.")

        # Replace values
        generated_text = template_sentence.format(**sentence_variables_values)

        # Get tokens
        entities = []
        if inserted_phenotype:
            for match in re.finditer(re.escape(inserted_phenotype), generated_text):
                entities.append(match.span())

        encoded = tokenizer(
            generated_text,
            max_length=512,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        tokens_offsets = encoded["offset_mapping"]

        # Extract tokens
        sentence_tokens = [generated_text[s:e] for s, e in tokens_offsets]
        sentence_tags = []

        for start, end in tokens_offsets:
            if start == end:
                sentence_tags.append("O") 
                continue
            
            tag = "O"
            for entity_start, entity_end in entities:
                # Si hay solapamiento (overlap)
                if start < entity_end and end > entity_start:
                    if start == entity_start:
                        tag = "B-SINTOMA"
                    else:
                        tag = "I-SINTOMA"
                    break

            sentence_tags.append(tag)

        for j in range(len(sentence_tags)):
            if sentence_tags[j] == "I-SINTOMA":
                if j == 0 or sentence_tags[j-1] == "O":
                    sentence_tags[j] = "B-SINTOMA"

        # Filter noise in tokens
        final_tokens = []
        final_tags = []
        
        for token, tag in zip(sentence_tokens, sentence_tags):
            if token.strip():
                final_tokens.append(token)
                final_tags.append(tag)

        if len(final_tokens) > 0:
            json_record = {
                "tokens": final_tokens,
                "ner_tags": final_tags
            }
            yield json.dumps(json_record, ensure_ascii=False) + "\n"

def generate_data_file(file_name, sentences_templates):
    phenotypes = get_phenotypes()

    # Data files
    with open(f"./output_data/{file_name}.jsonl", mode="w", encoding="utf-8") as output_file:

        # Go over phenotypes
        for phenotype in phenotypes:

            # Get sentences and annotations
            resultant_texts = generate_medical_history_sentences(phenotype["name"], sentences_templates)

            # Write lines of data
            for text in resultant_texts:
                output_file.write(text)

def generate_dataset(test_split, validation_split):
    sentences_templates = get_sentences_templates()

    # Calculate samples
    total_templates = len(sentences_templates)
    test_split_total_samples = int(test_split * total_templates)
    validation_split_total_samples = int(validation_split * total_templates)
    data_split_total_samples = total_templates - test_split_total_samples - validation_split_total_samples

    random.shuffle(sentences_templates)

    train_split_start, train_split_end = 0, data_split_total_samples
    train_sentences_templates = sentences_templates[train_split_start:train_split_end]

    validation_split_start, validation_split_end = train_split_end, train_split_end + validation_split_total_samples
    validation_sentences_templates = sentences_templates[validation_split_start: validation_split_end]

    test_split_start, test_split_end = validation_split_end, validation_split_end + test_split_total_samples
    test_sentences_templates = sentences_templates[test_split_start: test_split_end]

    # Generate train data
    generate_data_file("train_set", train_sentences_templates)

    # Generate validation data
    generate_data_file("validation_set", validation_sentences_templates)

    # Generate test data
    generate_data_file("test_set", test_sentences_templates)


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=0.2, type=float)
    parser.add_argument("--validation", default=0.1, type=float)

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(args.test, args.validation)