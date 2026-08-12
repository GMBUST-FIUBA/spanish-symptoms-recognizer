import argparse
import csv
import json
import random
import re

POSSIBLE_AGE = [
    "8 meses",
    "3 años",
    "5 años",
    "18 años",
    "25 años",
    "32 años",
    "45 años",
    "54 años", 
    "60 años",
    "72 años",
    "81 años",
]

POSSIBLE_BLOOD_PRESSURE = [
    "110/70 mmHg",
    "120/80 mmHg",
    "125/85 mmHg", 
    "130/80 mmHg",
    "140/90 mmHg",
    "150/95 mmHg",
]

POSSIBLE_HEART_RATE = [
    "60 lpm",
    "72 lpm",
    "85 lpm",
    "90 lpm", 
    "105 lpm",
    "110 lpm",
    "120 lpm",
]

POSSIBLE_TEMPERATURE = [
    "36.0 °C",
    "36.5 °C",
    "37.0 °C",
    "37.5 °C", 
    "38.0 °C",
    "38.5 °C",
    "39.2 °C"
]

POSSIBLE_OXYGEN_SATURATION = [
    "92%", "94%", "96%", "98%", "99%", "100%"
]

POSSIBLE_WEIGHT = [
    "55 kg",
    "62 kg",
    "70 kg",
    "78 kg",
    "85 kg",
    "92 kg",
]

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

POSSIBLE_SECTION_NUMBER = [
    "1", "2", "3", "4", "5", "6", "1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3", "3.1", "3.2", "I", "II", "III", "IV"
]

POSSIBLE_SECTION = [
    "Anamnesis", "Motivo de Consulta", "Enfermedad Actual", 
    "Antecedentes Personales", "Antecedentes Familiares", "Examen Físico", 
    "Signos Vitales", "Examen Neuropsiquiátrico", "Impresión Diagnóstica", 
    "Plan de Indicaciones", "Estudios Complementarios", "Evolución", "Epicrisis"
]

POSSIBLE_MEDICAMENT = [
    "Ranitidina", "Ondansetrón", "Ibuprofeno", "Dipirona", 
    "Dexketoprofeno", "Paracetamol", "Levonorgestrel", "Etinilestradiol", 
    "Omeprazol", "Aspirina", "Clonazepam", "Amoxicilina", "Losartán"
]

POSSIBLE_DOSE = [
    "40 mg", "8 mg EV en bolo", "1000 ml a 21 gotas/min", 
    "500 mg", "1 g", "cada 8 horas", "10 mg", "20 mg", 
    "50 mg", "1 ampolla", "2 comprimidos", "VO cada 12 hs"
]

POSSIBLE_STUDY = [
    "Tomografía Computada de Cerebro (TAC)", "Electrocardiograma (ECG)", 
    "Hemograma", "Laboratorio completo", "Ionograma", "Glucemia digital", 
    "Radiografía de tórax", "Resonancia Magnética", "Ecografía abdominal", 
    "Urocultivo", "Hepatograma", "Coagulograma"
]

POSSIBLE_INTENSITY = [
    "gran intensidad", "leve intensidad", "moderada intensidad", 
    "8/10 en escala analógica visual", "5/10", "severa", "muy severa", 
    "9/10", "2/10", "intensidad fluctuante", "baja intensidad", "intensidad intolerable"
]

POSSIBLE_LOCATION = [
    "holocraneana", "focalizada", "difusa", "frontal", "occipital", 
    "abdominal", "en fosa posterior", "parietal", "lumbar", "cervical", 
    "torácica", "en miembros inferiores", "hemicraneal"
]

POSSIBLE_CHARACTERISTIC = [
    "pulsátil", "inespecífica", "opresivo", "punzante", "rotatorio", 
    "en proyectil", "urente", "sordo", "intermitente", "constante", 
    "agudo", "crónico", "lacerante"
]

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

TEMPLATES_FILES = [
    "multple_phenotypes_templates.txt",
    "no_phenotypes_templates.txt",
    "single_phenotype_templates.txt",
    "boundary_templates.txt",
    "structural_templates.txt",
]

def add_noise_to_text(text, phenotypes):
    modified_text = text
    
    # Protect phenotypes
    for i, phenotype in enumerate(phenotypes):
        modified_text = modified_text.replace(phenotype, f"[[FENOTIPO_PROTEGIDO_{i}]]")

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

    for word, options in words_replacements.items():
        if random.random() < 0.6: 
            modified_text = re.sub(word, random.choice(options), modified_text, flags=re.IGNORECASE)

    modified_text = re.sub(r"\b(el|la|los|las|un|una)\s+", lambda m: "" if random.random() < 0.3 else m.group(0), modified_text, flags=re.IGNORECASE)
    modified_text = re.sub(r"\s+([.,;:!])", r"\1", modified_text)
    
    # Restore phenotypes
    for i, phenotype in enumerate(phenotypes):
        modified_text = modified_text.replace(f"[[FENOTIPO_PROTEGIDO_{i}]]", phenotype)
        
    return modified_text.strip()

def tokenize_text_for_training(text):
    return re.findall(r"[\wáéíóúñÁÉÍÓÚÑüÜ]+|[^\w\s]", text)

def tokenize_and_tag(text, phenotypes):
    tokens = tokenize_text_for_training(text)
    
    if not phenotypes:
        return tokens, ["O"] * len(tokens)

    phenotypes_tokens = [tokenize_text_for_training(p) for p in phenotypes]
    phenotypes_tokens.sort(key=len, reverse=True)
    
    tags = []
    i = 0
    
    while i < len(tokens):
        matched = False
        for p_tokens in phenotypes_tokens:
            if tokens[i:i+len(p_tokens)] == p_tokens:
                tags.append("B-SINTOMA")
                for _ in range(len(p_tokens) - 1):
                    tags.append("I-SINTOMA")
                i += len(p_tokens)
                matched = True
                break
        
        if not matched:
            tags.append("O")
            i += 1
            
    return tokens, tags

def generate_medical_history_sentences(main_phenotype_name, split_templates, split_phenotypes, templates_limit):

    phenotype_templates_limit = min(templates_limit, len(split_templates))
    selected_templates = random.sample(split_templates, phenotype_templates_limit)

    for template_sentence, template_variables in selected_templates:
        sentence_variables_values = {}
        inserted_phenotypes = []

        for var in template_variables:
            if "FENOTIPO" in var:
                if var == "FENOTIPO_1":
                    pheno = convert_first_char_to_lower_case(main_phenotype_name)
                else:
                    random_phenotype = random.choice(split_phenotypes)["name"]
                    pheno = convert_first_char_to_lower_case(random_phenotype)
                
                sentence_variables_values[var] = pheno
                inserted_phenotypes.append(pheno)
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
            elif "EDAD" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_AGE)
            elif "PRESION_SANGUINEA" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_BLOOD_PRESSURE)
            elif "FRECUENCIA_CARDIACA" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_HEART_RATE)
            elif "TEMPERATURA" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_TEMPERATURE)
            elif "PESO" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_WEIGHT)
            elif "SATURACION_OXIGENO" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_OXYGEN_SATURATION)
            elif "NUMERO_SECCION" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_SECTION_NUMBER)
            elif "SECCION" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_SECTION)
            elif "MEDICAMENTO" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_MEDICAMENT)
            elif "DOSIS" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_DOSE)
            elif "ESTUDIO" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_STUDY)
            elif "INTENSIDAD" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_INTENSITY)
            elif "LOCALIZACION" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_LOCATION)
            elif "CARACTERISTICA" in var:
                sentence_variables_values[var] = random.choice(POSSIBLE_CHARACTERISTIC)
            else:
                raise Exception(f"Found variable {var}")

        generated_text = template_sentence.format(**sentence_variables_values)
        
        # Add noise of abbreviations, errors, etc. to texts
        noisy_text = add_noise_to_text(generated_text, inserted_phenotypes)

        # Get tokens and tags
        final_tokens, final_tags = tokenize_and_tag(noisy_text, inserted_phenotypes)

        if len(final_tokens) > 0:
            json_record = {
                "tokens": final_tokens,
                "ner_tags": final_tags
            }
            yield json.dumps(json_record, ensure_ascii=False) + "\n"


def generate_data_file(file_name, sentences_templates, split_phenotypes):
    with open(f"./output_data/{file_name}.jsonl", mode="w", encoding="utf-8") as output_file:
        for phenotype in split_phenotypes:
            templates_to_use = random.randint(15, 20)

            resultant_texts = generate_medical_history_sentences(
                phenotype["name"], 
                sentences_templates, 
                split_phenotypes, 
                templates_to_use
            )
            for text in resultant_texts:
                output_file.write(text)

def generate_dataset(test_split, validation_split):
    phenotypes = get_phenotypes()

    all_templates = []

    # Load templates together
    for file_name in TEMPLATES_FILES:
        with open(f"input_texts/{file_name}", mode="r", encoding="utf-8") as file:
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
                all_templates.append((new_template.strip(), internal_variables))

    # Split phenotypes
    random.shuffle(phenotypes)
    total_phenotypes = len(phenotypes)
    test_count_phenos = int(test_split * total_phenotypes)
    val_count_phenos = int(validation_split * total_phenotypes)

    train_phenos_end = total_phenotypes - (test_count_phenos + val_count_phenos)
    val_phenos_end = total_phenotypes - test_count_phenos

    train_phenotypes = phenotypes[:train_phenos_end]
    validation_phenotypes = phenotypes[train_phenos_end:val_phenos_end]
    test_phenotypes = phenotypes[val_phenos_end:]

    # Gemerate datasets
    generate_data_file("train_set", all_templates, train_phenotypes)
    generate_data_file("validation_set", all_templates, validation_phenotypes)
    generate_data_file("test_set", all_templates, test_phenotypes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=0.2, type=float)
    parser.add_argument("--validation", default=0.1, type=float)
    args = parser.parse_args()

    generate_dataset(args.test, args.validation)