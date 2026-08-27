# Testing

## Introduction

This folder contains an End-to-End (E2E) evaluation pipeline designed to assess the performance of the phenotype recognizer. Models are benchmarked primarily using the micro/macro F1 score alongside complementary metrics, comparing different inference strategies and architectural approaches.

The models tested in this benchmark include:

- [es-hpo-phenotype-detector](https://huggingface.co/GonzaloMB/es-hpo-phenotype-detector): Trained internally using the scripts in `symptoms_recognizer/ner_model/training`.
- [HUMADEX/spanish_medical_ner](https://huggingface.co/HUMADEX/spanish_medical_ner)
- [medspaner/roberta-es-clinical-trials-umls-7sgs-ner](https://huggingface.co/medspaner/roberta-es-clinical-trials-umls-7sgs-ner)

The idea is not only to find the right combination of parameters for the clinical records analysis, but also to find the best way to parse the records to extract the most information and avoid the most noise possible. Several strategies where tested: sentence by sentence analysis, chunks analysis and sentences analyzed in specific contexts like *Background* or **. All of those strategies are explained in the next sections.

## Results

A table with the results of those models with different aggregation methods for sub-words and text parsing methods ordered by F1 score is presented:

| Test name | Model name | Aggregation strategy | TP | FP | FN | Precision | Recall | F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roberta-es-clinical-trials (agg: first, parsing: chunks-sentences) | roberta-es-clinical-trials-umls-7sgs-ner | first | 17 | 77 | 30 | 0.1809 | 0.3617 | 0.2411 |
| roberta-es-clinical-trials (agg: first, parsing: sections-sentences) | roberta-es-clinical-trials-umls-7sgs-ner | first | 17 | 80 | 30 | 0.1753 | 0.3617 | 0.2361 |
| roberta-es-clinical-trials (agg: first, parsing: sentences) | roberta-es-clinical-trials-umls-7sgs-ner | first | 17 | 80 | 30 | 0.1753 | 0.3617 | 0.2361 |
| roberta-es-clinical-trials (agg: average, parsing: chunks-sentences) | roberta-es-clinical-trials-umls-7sgs-ner | average | 14 | 60 | 33 | 0.1892 | 0.2979 | 0.2314 |
| roberta-es-clinical-trials (agg: average, parsing: sections-sentences) | roberta-es-clinical-trials-umls-7sgs-ner | average | 14 | 62 | 33 | 0.1842 | 0.2979 | 0.2276 |
| roberta-es-clinical-trials (agg: average, parsing: sentences) | roberta-es-clinical-trials-umls-7sgs-ner | average | 14 | 62 | 33 | 0.1842 | 0.2979 | 0.2276 |
| roberta-es-clinical-trials (agg: simple, parsing: chunks-sentences) | roberta-es-clinical-trials-umls-7sgs-ner | simple | 15 | 70 | 32 | 0.1765 | 0.3191 | 0.2273 |
| roberta-es-clinical-trials (agg: simple, parsing: sections-sentences) | roberta-es-clinical-trials-umls-7sgs-ner | simple | 15 | 73 | 32 | 0.1705 | 0.3191 | 0.2222 |
| roberta-es-clinical-trials (agg: simple, parsing: sentences) | roberta-es-clinical-trials-umls-7sgs-ner | simple | 15 | 73 | 32 | 0.1705 | 0.3191 | 0.2222 |
| base-nat-data (agg: first, parsing: chunks-sentences) | base-nat-data | first | 11 | 65 | 36 | 0.1447 | 0.2340 | 0.1789 |
| base-nat-data (agg: simple, parsing: chunks-sentences) | base-nat-data | simple | 11 | 65 | 36 | 0.1447 | 0.2340 | 0.1789 |
| base-nat-data (agg: first, parsing: sections-sentences) | base-nat-data | first | 11 | 65 | 36 | 0.1447 | 0.2340 | 0.1789 |
| base-nat-data (agg: first, parsing: sentences) | base-nat-data | first | 11 | 65 | 36 | 0.1447 | 0.2340 | 0.1789 |
| HUMADEX (agg: average, parsing: sentences) | HUMADEX | average | 10 | 69 | 37 | 0.1266 | 0.2128 | 0.1587 |
| HUMADEX (agg: average, parsing: sections-sentences) | HUMADEX | average | 10 | 69 | 37 | 0.1266 | 0.2128 | 0.1587 |
| HUMADEX (agg: first, parsing: sections-sentences) | HUMADEX | first | 10 | 73 | 37 | 0.1205 | 0.2128 | 0.1538 |
| HUMADEX (agg: first, parsing: sentences) | HUMADEX | first | 10 | 73 | 37 | 0.1205 | 0.2128 | 0.1538 |
| base-nat-data (agg: simple, parsing: sentences) | base-nat-data | simple | 9 | 62 | 38 | 0.1268 | 0.1915 | 0.1525 |
| base-nat-data (agg: simple, parsing: sections-sentences) | base-nat-data | simple | 9 | 62 | 38 | 0.1268 | 0.1915 | 0.1525 |
| HUMADEX (agg: average, parsing: chunks-sentences) | HUMADEX | average | 10 | 75 | 37 | 0.1176 | 0.2128 | 0.1515 |
| HUMADEX (agg: simple, parsing: sentences) | HUMADEX | simple | 7 | 41 | 40 | 0.1458 | 0.1489 | 0.1474 |
| HUMADEX (agg: simple, parsing: sections-sentences) | HUMADEX | simple | 7 | 41 | 40 | 0.1458 | 0.1489 | 0.1474 |
| HUMADEX (agg: first, parsing: chunks-sentences) | HUMADEX | first | 9 | 78 | 38 | 0.1034 | 0.1915 | 0.1343 |
| base-nat-data (agg: average, parsing: chunks-sentences) | base-nat-data | average | 7 | 53 | 40 | 0.1167 | 0.1489 | 0.1308 |
| base-nat-data (agg: average, parsing: sentences) | base-nat-data | average | 6 | 57 | 41 | 0.0952 | 0.1277 | 0.1091 |
| base-nat-data (agg: average, parsing: sections-sentences) | base-nat-data | average | 6 | 57 | 41 | 0.0952 | 0.1277 | 0.1091 |
| HUMADEX (agg: simple, parsing: chunks-sentences) | HUMADEX | simple | 3 | 32 | 44 | 0.0857 | 0.0638 | 0.0732 |
---

## Citations & References

If you use these baseline models or benchmark evaluations in your work, please cite the respective authors:

```bibtex
@article{app15105585,
  author  = {Sallauka, Rigon and Arioz, Umut and Rojc, Matej and Mlakar, Izidor},
  title   = {Weakly-Supervised Multilingual Medical NER for Symptom Extraction for Low-Resource Languages},
  journal = {Applied Sciences},
  volume  = {15},
  year    = {2025},
  number  = {10},
  article-number = {5585},
  url     = {[https://www.mdpi.com/2076-3417/15/10/5585](https://www.mdpi.com/2076-3417/15/10/5585)},
  issn    = {2076-3417},
  doi     = {10.3390/app15105585}
}

@article{campillosetal2024,
  author    = {Campillos-Llanos, Leonardo and Valverde-Mateos, Ana and Capllonch-Carri{\'o}n, Adri{\'a}n},
  title     = {Hybrid tool for semantic annotation and concept extraction of medical texts in Spanish},
  journal   = {BMC Bioinformatics},
  year      = {2024},
  publisher = {BioMed Central}
}
```