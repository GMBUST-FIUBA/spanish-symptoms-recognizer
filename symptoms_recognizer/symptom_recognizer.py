from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper
from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.chunks_sentences_parser import ChunkSentencesParser
from symptoms_recognizer.text_parser.full_text_parser import FullTextParser
from symptoms_recognizer.text_parser.sections_parser import SectionsSentencesParser
from symptoms_recognizer.text_parser.sentences_parser import SentencesParser

class PhenotypesRecognizer:
    def __init__(
        self, ner_model_path=None, ner_tokenizer_path=None, mapper_model_path=None,
        mapper_tokenizer_path=None, ontology=None, ontology_file_path=None,
        allowed_entity_groups=None, agg_strategy="simple", text_parser="sentences",
        phenotypes_model_type="ner",
        ner_api_provider=None, ner_api_model_name=None,
        map_api_provider=None, map_api_model_name=None,
    ):
        self.ner_model = PhenotypesDetector(
            model_path=ner_model_path, tokenizer_path=ner_tokenizer_path,
            allowed_entity_groups=allowed_entity_groups, agg_strategy=agg_strategy,
            phenotypes_model_type=phenotypes_model_type,
            api_provider=ner_api_provider, api_model_name=ner_api_model_name,
        )

        self.mapper = PhenotypeOntologyMapper(
            model_path=mapper_model_path, tokenizer_path=mapper_tokenizer_path,
            ontology=ontology, ontology_file_path=ontology_file_path,
            api_provider=map_api_provider, api_model_name=map_api_model_name
        )

        if text_parser == "chunks-sentences": self.text_parser = ChunkSentencesParser(max_chunk_tokens=384, overlap_sentences=1)
        elif text_parser == "sentences": self.text_parser = SentencesParser()
        elif text_parser == "sections-sentences": self.text_parser = SectionsSentencesParser()
        elif text_parser == "full-text": self.text_parser = FullTextParser()
        else: raise Exception("No known parser")

    def recognize(self, text: str) -> list[tuple[str, str]]:
        return self.text_parser.apply(text, self.ner_model)

    def map(self, phenotypes_with_context: list[tuple[str, str]]) -> list[str]:
        return self.mapper.map_phenotypes(phenotypes_with_context)

    def scan(self, text: str, only_results=False) -> list:
        symptoms_with_context = self.recognize(text)
        hpo_codes = self.map(symptoms_with_context)

        if only_results:
            return hpo_codes

        return [(item[0], hpo_code) for item, hpo_code in zip(symptoms_with_context, hpo_codes) if hpo_code != "None"]