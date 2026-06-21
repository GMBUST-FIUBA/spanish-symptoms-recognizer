import argparse
import csv
import random
import re

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

            sentences_templates.append((new_template, internal_variables))

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

def generate_medical_history_sentences(phenotype_name, sentences_templates, samples):

    chosen_elements = set()

    for _ in range(samples):
        # Choose a new element
        template_number = random.randrange(0, len(sentences_templates))
        while template_number in chosen_elements:
            template_number = random.randrange(0, len(sentences_templates))
        chosen_elements.add(template_number)

        # Get template
        template_chosen = sentences_templates[template_number]
        template_sentence = template_chosen[0]
        template_variables = template_chosen[1]

        # Iterate over variables of sentence
        sentence_variables_values = {}
        for var in template_variables:
            if "FENOTIPO" in var:
                sentence_variables_values[var] = convert_first_char_to_lower_case(phenotype_name)
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
            else:
                raise Exception("Unknown variable found.")

        # Replace values
        yield template_sentence.format(**sentence_variables_values)

def generate_dataset(samples):
    phenotypes = get_phenotypes()
    sentences_templates = get_sentences_templates()

    if len(sentences_templates) < samples:
        raise Exception("Not enough templates for uniques sentences")

    with open("./output_data/data.txt", mode="w", encoding="utf-8") as output_file:
        for phenotype in phenotypes:

            resultant_texts = generate_medical_history_sentences(phenotype["name"], sentences_templates, samples)
            for text in resultant_texts:
                output_file.write(text)


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=3, type=int)

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(args.samples)