# Testing

## Introduction

This folder contains an End-to-End (E2E) evaluation pipeline designed to assess the performance of the phenotype recognizer. Models are benchmarked primarily using the micro/macro F1 score alongside complementary metrics, comparing different inference strategies and architectural approaches.

The models tested in this benchmark include:

- [es-hpo-phenotype-detector](https://huggingface.co/GonzaloMB/es-hpo-phenotype-detector): Trained internally using the scripts in `symptoms_recognizer/ner_model/training`.
- [HUMADEX/spanish_medical_ner](https://huggingface.co/HUMADEX/spanish_medical_ner)
- [medspaner/roberta-es-clinical-trials-umls-7sgs-ner](https://huggingface.co/medspaner/roberta-es-clinical-trials-umls-7sgs-ner)

The idea is not only to find the right combination of parameters for the clinical records analysis, but also to find the best way to parse the records to extract the most information and avoid the most noise possible. Several strategies where tested: sentence by sentence analysis, chunks analysis and sentences analyzed in specific contexts like *Background* (Antecedentes in spanish) or *Current illness* (Enfermedad actual). All of those strategies are explained in the next sections.

## Results

### First test

A table with the results of those models with different aggregation methods for sub-words and text parsing methods ordered by F1 score is presented:

| Test name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 | Map TP | Map FP | Map FN | Map Prec | Map Rec | Final F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: first, parsing: chunks-sentences) | 14 | 484 | 45 | 0.0281 | 0.2373 | 0.0503 | 22 | 88 | 37 | 0.2000 | 0.3729 | 0.2604 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: first, parsing: sentences) | 15 | 474 | 44 | 0.0307 | 0.2542 | 0.0547 | 22 | 92 | 37 | 0.1930 | 0.3729 | 0.2543 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: first, parsing: sections-sentences) | 15 | 458 | 44 | 0.0317 | 0.2542 | 0.0564 | 22 | 92 | 37 | 0.1930 | 0.3729 | 0.2543 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: simple, parsing: chunks-sentences) | 34 | 593 | 25 | 0.0542 | 0.5763 | 0.0991 | 20 | 83 | 39 | 0.1942 | 0.3390 | 0.2469 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: simple, parsing: sentences) | 35 | 585 | 24 | 0.0565 | 0.5932 | 0.1031 | 20 | 86 | 39 | 0.1887 | 0.3390 | 0.2424 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: simple, parsing: sections-sentences) | 35 | 571 | 24 | 0.0578 | 0.5932 | 0.1053 | 20 | 86 | 39 | 0.1887 | 0.3390 | 0.2424 |
| base-nat-data (agg: first, parsing: sections-sentences) | 11 | 136 | 48 | 0.0748 | 0.1864 | 0.1068 | 15 | 61 | 44 | 0.1974 | 0.2542 | 0.2222 |
| base-nat-data (agg: first, parsing: chunks-sentences) | 10 | 134 | 49 | 0.0694 | 0.1695 | 0.0985 | 15 | 61 | 44 | 0.1974 | 0.2542 | 0.2222 |
| base-nat-data (agg: first, parsing: sentences) | 11 | 136 | 48 | 0.0748 | 0.1864 | 0.1068 | 15 | 61 | 44 | 0.1974 | 0.2542 | 0.2222 |
| base-nat-data (agg: simple, parsing: chunks-sentences) | 27 | 124 | 32 | 0.1788 | 0.4576 | 0.2571 | 14 | 62 | 45 | 0.1842 | 0.2373 | 0.2074 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: average, parsing: chunks-sentences) | 14 | 428 | 45 | 0.0317 | 0.2373 | 0.0559 | 15 | 74 | 44 | 0.1685 | 0.2542 | 0.2027 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: average, parsing: sections-sentences) | 15 | 412 | 44 | 0.0351 | 0.2542 | 0.0617 | 15 | 77 | 44 | 0.1630 | 0.2542 | 0.1987 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: average, parsing: sentences) | 15 | 413 | 44 | 0.0350 | 0.2542 | 0.0616 | 15 | 77 | 44 | 0.1630 | 0.2542 | 0.1987 |
| base-nat-data (agg: simple, parsing: sentences) | 25 | 128 | 34 | 0.1634 | 0.4237 | 0.2358 | 12 | 59 | 47 | 0.1690 | 0.2034 | 0.1846 |
| base-nat-data (agg: simple, parsing: sections-sentences) | 25 | 128 | 34 | 0.1634 | 0.4237 | 0.2358 | 12 | 59 | 47 | 0.1690 | 0.2034 | 0.1846 |
| HUMADEX (agg: average, parsing: sections-sentences) | 14 | 530 | 45 | 0.0257 | 0.2373 | 0.0464 | 12 | 71 | 47 | 0.1446 | 0.2034 | 0.1690 |
| HUMADEX (agg: average, parsing: sentences) | 14 | 545 | 45 | 0.0250 | 0.2373 | 0.0453 | 12 | 71 | 47 | 0.1446 | 0.2034 | 0.1690 |
| HUMADEX (agg: average, parsing: chunks-sentences) | 14 | 477 | 45 | 0.0285 | 0.2373 | 0.0509 | 12 | 75 | 47 | 0.1379 | 0.2034 | 0.1644 |
| HUMADEX (agg: first, parsing: sentences) | 13 | 659 | 46 | 0.0193 | 0.2203 | 0.0356 | 12 | 76 | 47 | 0.1364 | 0.2034 | 0.1633 |
| HUMADEX (agg: first, parsing: sections-sentences) | 13 | 630 | 46 | 0.0202 | 0.2203 | 0.0370 | 12 | 76 | 47 | 0.1364 | 0.2034 | 0.1633 |
| base-nat-data (agg: average, parsing: chunks-sentences) | 10 | 120 | 49 | 0.0769 | 0.1695 | 0.1058 | 9 | 51 | 50 | 0.1500 | 0.1525 | 0.1513 |
| HUMADEX (agg: simple, parsing: sections-sentences) | 8 | 1081 | 51 | 0.0073 | 0.1356 | 0.0139 | 8 | 42 | 51 | 0.1600 | 0.1356 | 0.1468 |
| HUMADEX (agg: simple, parsing: sentences) | 8 | 1130 | 51 | 0.0070 | 0.1356 | 0.0134 | 8 | 42 | 51 | 0.1600 | 0.1356 | 0.1468 |
| HUMADEX (agg: first, parsing: chunks-sentences) | 13 | 626 | 46 | 0.0203 | 0.2203 | 0.0372 | 11 | 81 | 48 | 0.1196 | 0.1864 | 0.1457 |
| base-nat-data (agg: average, parsing: sentences) | 11 | 119 | 48 | 0.0846 | 0.1864 | 0.1164 | 8 | 55 | 51 | 0.1270 | 0.1356 | 0.1311 |
| base-nat-data (agg: average, parsing: sections-sentences) | 11 | 119 | 48 | 0.0846 | 0.1864 | 0.1164 | 8 | 55 | 51 | 0.1270 | 0.1356 | 0.131 |
| HUMADEX (agg: simple, parsing: chunks-sentences) | 1 | 1219 | 58 | 0.0008 | 0.0169 | 0.0016 | 5 | 31 | 54 | 0.1389 | 0.0847 | 0.1053 |

As it can be seen, the models trained on data with little to no AI involvement have better overall predictions scores, being [Medspanner's model](https://huggingface.co/medspaner/roberta-es-clinical-trials-umls-7sgs-ner) the best one. However it is worth noting that the models have an intrinsic hallucination problem, which is evident in the best models where sometimes there is a 4:1 ratio between false positives (hallucinations) and true positives at the end of the end of the pipeline, but a ratio of almost 1:35 at the end of the NER stage. And finally what is inferred through this results is that the aggregation method known as *first* works better than the other ones no matter the model used, and *chunks of sentences* text parsing is better in general.

### Second test

What was hypothesised was that adjusting the minimum distance a vector must have to an HPO type could help reduce the false positives and it was changed from 0.2 to 0.1 using de cosine distance. The results are shown as follows:

| Test name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 | Map TP | Map FP | Map FN | Map Prec | Map Rec | Final F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: first, parsing: sections-sentences) | 15 | 458 | 44 | 0.0317 | 0.2542 | 0.0564 | 18 | 27 | 41 | 0.4000 | 0.3051 | 0.3462 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: first, parsing: sentences) | 15 | 474 | 44 | 0.0307 | 0.2542 | 0.0547 | 18 | 27 | 41 | 0.4000 | 0.3051 | 0.3462 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: first, parsing: chunks-sentences) | 14 | 484 | 45 | 0.0281 | 0.2373 | 0.0503 | 18 | 28 | 41 | 0.3913 | 0.3051 | 0.3429 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: simple, parsing: sentences) | 35 | 585 | 24 | 0.0565 | 0.5932 | 0.1031 | 17 | 24 | 42 | 0.4146 | 0.2881 | 0.3400 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: simple, parsing: sections-sentences) | 35 | 571 | 24 | 0.0578 | 0.5932 | 0.1053 | 17 | 24 | 42 | 0.4146 | 0.2881 | 0.3400 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: simple, parsing: chunks-sentences) | 34 | 593 | 25 | 0.0542 | 0.5763 | 0.0991 | 17 | 25 | 42 | 0.4048 | 0.2881 | 0.3366 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: average, parsing: sentences) | 15 | 413 | 44 | 0.0350 | 0.2542 | 0.0616 | 12 | 25 | 47 | 0.3243 | 0.2034 | 0.2500 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: average, parsing: sections-sentences) | 15 | 412 | 44 | 0.0351 | 0.2542 | 0.0617 | 12 | 25 | 47 | 0.3243 | 0.2034 | 0.2500 |
| base-nat-data (agg: simple, parsing: chunks-sentences) | 27 | 124 | 32 | 0.1788 | 0.4576 | 0.2571 | 10 | 11 | 49 | 0.4762 | 0.1695 | 0.2500 |
| roberta-es-clinical-trials-umls-7sgs-ner (agg: average, parsing: chunks-sentences) | 14 | 428 | 45 | 0.0317 | 0.2373 | 0.0559 | 12 | 26 | 47 | 0.3158 | 0.2034 | 0.2474 |
| base-nat-data (agg: first, parsing: chunks-sentences) | 10 | 134 | 49 | 0.0694 | 0.1695 | 0.0985 | 10 | 13 | 49 | 0.4348 | 0.1695 | 0.2439 |
| base-nat-data (agg: first, parsing: sentences) | 11 | 136 | 48 | 0.0748 | 0.1864 | 0.1068 | 10 | 14 | 49 | 0.4167 | 0.1695 | 0.2410 |
| base-nat-data (agg: first, parsing: sections-sentences) | 11 | 136 | 48 | 0.0748 | 0.1864 | 0.1068 | 10 | 14 | 49 | 0.4167 | 0.1695 | 0.2410 |
| base-nat-data (agg: simple, parsing: sentences) | 25 | 128 | 34 | 0.1634 | 0.4237 | 0.2358 | 9 | 11 | 50 | 0.4500 | 0.1525 | 0.2278 |
| base-nat-data (agg: simple, parsing: sections-sentences) | 25 | 128 | 34 | 0.1634 | 0.4237 | 0.2358 | 9 | 11 | 50 | 0.4500 | 0.1525 | 0.2278 |
| base-nat-data (agg: average, parsing: chunks-sentences) | 10 | 120 | 49 | 0.0769 | 0.1695 | 0.1058 | 8 | 11 | 51 | 0.4211 | 0.1356 | 0.2051 |
| base-nat-data (agg: average, parsing: sentences) | 11 | 119 | 48 | 0.0846 | 0.1864 | 0.1164 | 7 | 12 | 52 | 0.3684 | 0.1186 | 0.1795 |
| base-nat-data (agg: average, parsing: sections-sentences) | 11 | 119 | 48 | 0.0846 | 0.1864 | 0.1164 | 7 | 12 | 52 | 0.3684 | 0.1186 | 0.1795 |
| HUMADEX (agg: first, parsing: sentences) | 13 | 659 | 46 | 0.0193 | 0.2203 | 0.0356 | 6 | 25 | 53 | 0.1935 | 0.1017 | 0.1333 |
| HUMADEX (agg: first, parsing: sections-sentences) | 13 | 630 | 46 | 0.0202 | 0.2203 | 0.0370 | 6 | 25 | 53 | 0.1935 | 0.1017 | 0.1333 |
| HUMADEX (agg: average, parsing: sentences) | 14 | 545 | 45 | 0.0250 | 0.2373 | 0.0453 | 6 | 28 | 53 | 0.1765 | 0.1017 | 0.1290 |
| HUMADEX (agg: average, parsing: sections-sentences) | 14 | 530 | 45 | 0.0257 | 0.2373 | 0.0464 | 6 | 28 | 53 | 0.1765 | 0.1017 | 0.1290 |
| HUMADEX (agg: average, parsing: chunks-sentences) | 14 | 477 | 45 | 0.0285 | 0.2373 | 0.0509 | 6 | 30 | 53 | 0.1667 | 0.1017 | 0.1263 |
| HUMADEX (agg: simple, parsing: chunks-sentences) | 1 | 1219 | 58 | 0.0008 | 0.0169 | 0.0016 | 4 | 12 | 55 | 0.2500 | 0.0678 | 0.1067 |
| HUMADEX (agg: first, parsing: chunks-sentences) | 13 | 626 | 46 | 0.0203 | 0.2203 | 0.0372 | 5 | 30 | 54 | 0.1429 | 0.0847 | 0.1064 |
| HUMADEX (agg: simple, parsing: sentences) | 8 | 1130 | 51 | 0.0070 | 0.1356 | 0.0134 | 3 | 17 | 56 | 0.1500 | 0.0508 | 0.0759 |
| HUMADEX (agg: simple, parsing: sections-sentences) | 8 | 1081 | 51 | 0.0073 | 0.1356 | 0.0139 | 3 | 17 | 56 | 0.1500 | 0.0508 | 0.0759 |

The results validate the hypothesis and confirm that a tighter distance between vectors helps by almost halving the number of hallucinations at the end of the pipeline with the false positives but at the cost of not capturing some phenotypes. This means that the hallucination problem is made by elements in the text that the NER model considers as phenotypes but are not really that close semantically to the HPO phenotypes (and this includes both noise and some relevant medical concepts that have no interest here), and that some phenotypes that must be captured are not that close semantically to the HPO names using the selected embedding and therefore are not captured at the end of the mapping stage.

### Third test

The next test was made to test an LLM in order to detect phenotypes in the clinical records. The reluctance to test this models was due to their non-deterministic results. So, as a proof of concept it was used a Gemini 3.6-flash free trial and the results were promosing, so a more detailed test was made in a model with more requests available as the Gemini 3.1 Flash Lite model:

| Model name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 score | Map TP | Map FP | Map FN | Map Prec | Map Rec | Map F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 3.1-flash-lite | 50 | 59 | 9 | 0.4587 | 0.8475 | 0.5952 | 17 | 17 | 42 | 0.5 | 0.2881 | 0.3656 |

This is a major breakthrough in terms of performance since not only where almost all phenotypes detected (84% where detected) but also the hallucinations where reduced by almost 87%. Also, this implies that an upgrade in the LLMs used will more likely upgrade the detection of phenotypes. However an improvement needs to be made in the mapping stage since the mapper model used for this stage, the [ClinLinker-KB-GP](https://huggingface.co/ICB-UMA/ClinLinker-KB-GP), can't appropiately match the terms acquired by the NER model to the HPO phenotypes.

After this a new test was developed in order to get more data. This was tested using the Gemini models Gemini 3.5 Flash, Gemini 3.1 Flash Lite and Gemini 3 Flash (preview stage). The results are shown as follows:

| Model name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 score | Map TP | Map FP | Map FN | Map Prec | Map Rec | Map F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini - gemini-3.5-flash | 52 | 40 | 7 | 0.5652 | 0.8814 | 0.6887 | 17 | 11 | 42 | 0.6071 | 0.2881 | 0.3908 |
| Gemini - gemini-3.1-flash-lite | 51 | 62 | 8 | 0.4513 | 0.8644 | 0.5930 | 17 | 13 | 42 | 0.5667 | 0.2881 | 0.3820 |
| Gemini - gemini-3-flash-preview | 48 | 82 | 11  | 0.3692 | 0.8136 | 0.5079 | 16 | 13 | 43 | 0.5517 | 0.2712 | 0.3636 |

As can be shown in the table above, the result is the best so far overall but specially at the NER stage where the false positives are at their lowest. More models will be used in the future in order to determine the best ones.


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