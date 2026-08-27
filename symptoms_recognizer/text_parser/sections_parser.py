import spacy
from medspacy.section_detection import SectionRule
from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.text_parser_interface import HistoryRecordParser

class SectionsSentencesParser(HistoryRecordParser):
    def __init__(self):
        self.text_nlp = spacy.blank("es")

        self.text_nlp.add_pipe("sentencizer")

        self.sectionizer = self.text_nlp.add_pipe("medspacy_sectionizer")

        section_rules = [
            # Motives for consulting
            SectionRule(literal="Motivo de consulta", category="motivo_consulta", pattern=[{"LOWER": "motivo"}, {"LOWER": "de"}, {"LOWER": "consulta"}]),
            SectionRule(literal="Razón de consulta", category="motivo_consulta", pattern=[{"LOWER": "razón"}, {"LOWER": "de"}, {"LOWER": "consulta"}]),
            SectionRule(literal="Problema actual", category="motivo_consulta", pattern=[{"LOWER": "problema"}, {"LOWER": "actual"}]),

            # Background
            SectionRule(literal="Antecedentes personales", category="antecedentes", pattern=[{"LOWER": "antecedentes"}, {"LOWER": "personales"}]),
            SectionRule(literal="Antecedentes personales patológicos", category="antecedentes", pattern=[{"LOWER": "antecedentes"}, {"LOWER": "personales"}, {"LOWER": "patológicos"}]),
            SectionRule(literal="Historia médica", category="antecedentes", pattern=[{"LOWER": "historia"}, {"LOWER": "médica"}]),
            SectionRule(literal="Antecedentes", category="antecedentes", pattern=[{"LOWER": "antecedentes"}]),

            # Current illnesses
            SectionRule(literal="Enfermedad actual", category="enfermedad_actual", pattern=[{"LOWER": "enfermedad"}, {"LOWER": "actual"}]),

            # Physical exam
            SectionRule(literal="Examen físico", category="examen_fisico", pattern=[{"LOWER": "examen"}, {"LOWER": "físico"}]),

            # Other studies
            SectionRule(literal="Estudios complementarios", category="estudios_complementarios", pattern=[{"LOWER": "estudios"}, {"LOWER": "complementarios"}]),

            # Final section
            SectionRule(literal="Impresión Diagnóstica y Plan", category="seccion_final", pattern=[{"LOWER": "impresión"}, {"LOWER": "diagnóstica"}, {"LOWER": "y"}, {"LOWER": "plan"}]),
            SectionRule(literal="Impresión Diagnóstica", category="seccion_final", pattern=[{"LOWER": "impresión"}, {"LOWER": "diagnóstica"}]),
        ]
        self.sectionizer.add(section_rules)

        self.target_sections = {
            "motivo_consulta", 
            "enfermedad_actual", 
            "antecedentes",
            "examen_fisico",
            "estudios_complementarios",
            "seccion_final"
        }

    def apply(self, text: str, ner_model: PhenotypesDetector) -> list[str]:
        doc = self.text_nlp(text)
        
        raw_phenotypes = []

        for section in doc._.sections:
            if section.category in self.target_sections:
                start, end = section.body_span
                
                if start == end:
                    continue
                
                real_body_span = doc[start:end]

                for sent in real_body_span.sents:
                    sentence_text = sent.text.strip()

                    if not sentence_text:
                        continue

                    sentence_results = ner_model.detect_phenotypes(sentence_text)
                    raw_phenotypes.extend(sentence_results)

        seen = set()
        unique_phenotypes = []
        for p in raw_phenotypes:
            normalized = p.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_phenotypes.append(p.strip())

        return unique_phenotypes