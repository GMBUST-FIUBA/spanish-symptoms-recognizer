from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper
from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.chunks_sentences_parser import ChunkSentencesParser
from symptoms_recognizer.text_parser.sections_parser import SectionsSentencesParser
from symptoms_recognizer.text_parser.sentences_parser import SentencesParser

import spacy

class PhenotypesRecognizer:
    def __init__(
        self, 
        ner_model_path=None,
        ner_tokenizer_path=None,
        mapper_model_path=None,
        mapper_tokenizer_path=None,
        ontology=None,
        ontology_file_path=None,
        allowed_entity_groups=None,
        agg_strategy="simple",
        text_parser="sentences-parser"
    ):

        # Initialize NER model
        self.ner_model = PhenotypesDetector(
            model_path=ner_model_path, 
            tokenizer_path=ner_tokenizer_path,
            allowed_entity_groups=allowed_entity_groups,
            agg_strategy=agg_strategy
        )

        # Initialize mapper
        self.mapper = PhenotypeOntologyMapper(
            model_path=mapper_model_path,
            tokenizer_path=mapper_tokenizer_path,
            ontology=ontology,
            ontology_file_path=ontology_file_path
        )

        # Initialize text splitter
        if text_parser == "chunks-sentences":
            self.text_parser = ChunkSentencesParser(max_chunk_tokens=384, overlap_sentences=1)
        elif text_parser == "sentences":
            self.text_parser = SentencesParser()
        elif text_parser == "sections-sentences":
            self.text_parser = SectionsSentencesParser()
        else:
            raise Exception("No known parser")

    def recognize(self, text: str) -> list[str]:
        return self.text_parser.apply(text, self.ner_model)

    def map(self, phenotypes_list: list[str]) -> list[str]:
        return self.mapper.map_phenotypes(phenotypes_list)

    def scan(self, text: str, only_results=False) -> list:
        symptoms_list = self.recognize(text)
        hpo_codes = self.map(symptoms_list)

        if only_results:
            return hpo_codes

        return [(symptom, hpo_code) for symptom, hpo_code in zip(symptoms_list, hpo_codes) if hpo_code != "None"]