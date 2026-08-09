import argparse
import csv
import json
import random
import re

def convert_first_char_to_lower_case(text: str):
    return text.replace(text[0], text[0].lower(), 1)

def get_phenotypes():
    phenotypes = []
    with open("input_texts/hp-es.babelon.tsv", mode="r", encoding="utf-8") as file:
        file_reader = csv.DictReader(file, delimiter="\t")
        next(file_reader) # To avoid the "All" category

        for row in file_reader:
            phenotypes.append({
                "code" : row["subject_id"],
                "name" : row["translation_value"]
            })
    return phenotypes

def get_sentences_templates():
    sentences_templates = []
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

def add_noise_to_text(text, phenotype):
    PHENOTYPE_PLACEHOLDER = "[[FENOTIPO_PROTEGIDO]]"

    if phenotype:
        modified_text = text.replace(phenotype, PHENOTYPE_PLACEHOLDER)
    else:
        modified_text = text

    words_replacements = {
        r"\bpaciente\b": ["pte", "pte.", "pac."],
        r"\bdiagnóstico\b": ["dx", "diag", "dg"],
        r"\bdiagnosticado\b": ["dx", "diag", "dg"],
        r"\bcon\b": ["c/"],
        r"\bsin\b": ["s/"],
        r"\btratamiento\b": ["tto", "tx"],
        r"\bantecedentes\b": ["ant", "ant."],
        r"\bhistoria clínica\b": ["hc", "h.c."],
        r"\bevolución\b": ["ev", "evol"],
    }

    # Replace words for informal terms

    ## For certain types of words
    for word, options in words_replacements.items():
        if random.random() < 0.6: 
            modified_text = re.sub(word, random.choice(options), modified_text, flags=re.IGNORECASE)

    ## Random erasing of articles
    modified_text = re.sub(r"\b(el|la|los|las|un|una)\s+", lambda m: "" if random.random() < 0.3 else m.group(0), modified_text, flags=re.IGNORECASE)

    ## Put random signs
    modified_text = re.sub(r"\s+([.,;:!])", r"\1", modified_text)
    
    # Restore phenotype
    if phenotype:
        texto_final = modified_text.replace(PHENOTYPE_PLACEHOLDER, phenotype)
    else:
        texto_final = modified_text
        
    return texto_final.strip()

def tokenize_and_tag(text, entity):
    """
    Tokeniza por palabras y signos de puntuación, asignando etiquetas BIO.
    Soporta el caso de oraciones sin entidad (entity = "").
    """
    tokens = re.findall(r"[\wáéíóúñÁÉÍÓÚÑüÜ]+|[^\w\s]", text)
    
    # Si es una oración nula, todos los tokens son "O"
    if not entity:
        return tokens, ["O"] * len(tokens)
        
    entity_tokens = re.findall(r"[\wáéíóúñÁÉÍÓÚÑüÜ]+|[^\w\s]", entity)
    tags = []
    i = 0
    
    while i < len(tokens):
        if tokens[i:i+len(entity_tokens)] == entity_tokens:
            tags.append("B-SINTOMA")
            for _ in range(len(entity_tokens) - 1):
                tags.append("I-SINTOMA")
            i += len(entity_tokens)
        else:
            tags.append("O")
            i += 1
            
    return tokens, tags

def generate_medical_history_sentences(phenotype_name, sentences_templates):
    for template_sentence, template_variables in sentences_templates:
        sentence_variables_values = {}
        inserted_phenotype = ""

        for var in template_variables:
            if "FENOTIPO" in var:
                inserted_phenotype = convert_first_char_to_lower_case(phenotype_name)
                sentence_variables_values[var] = inserted_phenotype
            elif "HORA" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_TIMES)
            elif "LUGAR_MEDICO" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_MEDICAL_PLACES)
            elif "ETAPA_DIA" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_DAY_TIMES)
            elif "ACOMPANANTE" in var:
                companion = random.choice(POSSIBLE_COMPANIONS)
                if "ACOMPANANTE_INICIO" in var:
                    companion = convert_first_char_to_lower_case(companion)
                sentence_variables_values[var] = companion
            elif "TIEMPO_PASADO" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_PASSED_TIME)
            elif "SEXO_PACIENTE" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_PATIENT_SEX)

        generated_text = template_sentence.format(**sentence_variables_values)
        
        # Add noise of abbreviations, errors, etc. to texts
        noisy_text = add_noise_to_text(generated_text, inserted_phenotype)

        # Get tokens and tags
        final_tokens, final_tags = tokenize_and_tag(noisy_text, inserted_phenotype)

        if len(final_tokens) > 0:
            json_record = {
                "tokens": final_tokens,
                "ner_tags": final_tags
            }
            yield json.dumps(json_record, ensure_ascii=False) + "\n"

def generate_data_file(file_name, sentences_templates, phenotypes):
    with open(f"./output_data/{file_name}.jsonl", mode="w", encoding="utf-8") as output_file:
        for phenotype in phenotypes:
            resultant_texts = generate_medical_history_sentences(phenotype["name"], sentences_templates)
            for text in resultant_texts:
                output_file.write(text)

def generate_dataset(test_split, validation_split):
    sentences_templates = get_sentences_templates()
    phenotypes = get_phenotypes()

    # Partition templates
    random.shuffle(sentences_templates)
    total_sentences_templates = len(sentences_templates)
    test_split_total_templates = int(test_split * total_sentences_templates)
    validation_split_total_templates = int(validation_split * total_sentences_templates)

    train_templates = sentences_templates[:-(test_split_total_templates + validation_split_total_templates)]
    validation_templates = sentences_templates[-(test_split_total_templates + validation_split_total_templates):-test_split_total_templates]
    test_templates = sentences_templates[-test_split_total_templates:]

    # Partition phenotypes
    random.shuffle(phenotypes)
    total_phenotypes = len(phenotypes)
    test_split_total_phenotypes = int(test_split * total_phenotypes)
    validation_split_total_phenotypes = int(validation_split * total_phenotypes)
    
    train_phenotypes = phenotypes[:-(test_split_total_phenotypes + validation_split_total_phenotypes)]
    validation_phenotypes = phenotypes[-(test_split_total_phenotypes + validation_split_total_phenotypes):-test_split_total_phenotypes]
    test_phenotypes = phenotypes[-test_split_total_phenotypes:]

    # Generate data files
    generate_data_file("train_set", train_templates, train_phenotypes)
    generate_data_file("validation_set", validation_templates, validation_phenotypes)
    generate_data_file("test_set", test_templates, test_phenotypes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=0.2, type=float)
    parser.add_argument("--validation", default=0.1, type=float)
    args = parser.parse_args()

    generate_dataset(args.test, args.validation)