# Testing

## Introduction

This folder contains an End-to-End (E2E) evaluation pipeline designed to assess the performance of the phenotype recognizer. Models are benchmarked primarily using the micro/macro F1 score alongside complementary metrics, comparing different inference strategies and architectural approaches.

The models tested in this benchmark include:

- [es-hpo-phenotype-detector](https://huggingface.co/GonzaloMB/es-hpo-phenotype-detector): Trained internally using the scripts in `symptoms_recognizer/ner_model/training`.
- [HUMADEX/spanish_medical_ner](https://huggingface.co/HUMADEX/spanish_medical_ner)
- [medspaner/roberta-es-clinical-trials-umls-7sgs-ner](https://huggingface.co/medspaner/roberta-es-clinical-trials-umls-7sgs-ner)

The idea is not only to find the right combination of parameters for the clinical records analysis, but also to find the best way to parse the records to extract the most information and avoid the most noise possible. Several strategies where tested: sentence by sentence analysis, chunks analysis and sentences analyzed in specific contexts like *Background* (Antecedentes in spanish) or *Current illness* (Enfermedad actual). All of those strategies are explained in the next sections.

## Results

### First test: NER models test

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

### Second test: Minimum vectors cosine distance lowered

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

### Third test: NER stage done by LLM's

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

### Fourth test: Mapping HPO codes with RAG and context using LLM's

Since the mapping stage was very inefficient at mapping phenotypes to their right codes, then a new idea was developed in order to better the results. What was found after analyzing the model was that even though the cosine distance is good at detecting similarities between phenotypes and HPO terms, if only the phenotype is sent to the mapping stage then a several possible codes can be found for a found. For example, if a the phenotype is "mocos" (mucus) then some possible HPO terms may be for "secreción anormal de moco nasal" (abnormal nasal mucus secretion) or "Rinorrea" (Rhinorrhea) since all the information provided is "mocos". This can be aided by the introduction of some context for the phenotype in the analysis. Therefore what is decided is that the NER stage must also separate the context of the phenotype detected and pass it to the mapping stage to process it, and to accomplish this task of reasoning using the context an AI model will be used.

Another point to consider was how to guess the possible HPO codes. Since the HPO currently has over 18.000 terms if all of them are made of words that can be tokenized in 6 tokens (something that isn't really tru but serves as an example) then a total of 108.000 tokens need to be analyzed on each call to the AI in order to correctly assess the HPO code to use, which is not only impractical since a lot of LLMs that can fit this context nowadays are considered among the best but it is also very costly computationally and econnomically speaking. Therefore a RAG approach was decided, where the *ClinLinker* will still be used in order to define the closest HPO terms to the detected phenotype using the cosine distance but the top-K closest phenotypes (where K is a natural number) will be sent to the AI model in order to consider them as possible candidates to the HPO code.

This approach was implemented on the code and here are the results for K=5 and K=10 respectively:

| Model name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 score | Map TP | Map FP | Map FN | Map Prec | Map Rec | Final F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini - NER: gemini-3.5-flash - MAP: gemini-3.5-flash | 52 | 56 | 7 | 0.4815 | 0.8814 | 0.6228 | 34 | 45 | 25 | 0.4304 | 0.5763 | 0.4928 |
| Gemini - NER: gemini-3.1-flash-lite - MAP: gemini-3.5-flash | 51 | 55 | 8 | 0.4811 | 0.8644 | 0.6182 | 32 | 46 | 27 | 0.4103 | 0.5424 | 0.4672 |
| Gemini - NER: gemini-3-flash-preview - MAP: gemini-3.5-flash | 52 | 93 | 7 | 0.3586 | 0.8814 | 0.5098 | 32 | 65 | 27 | 0.3299 | 0.5424 | 0.4103 |


| Model name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 score | Map TP | Map FP | Map FN | Map Prec | Map Rec | Final F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini - NER: gemini-3.5-flash - MAP: gemini-3.5-flash | 49 | 66 | 10 | 0.4261 | 0.8305 | 0.5632 | 33 | 47 | 26 | 0.4125 | 0.5593 | 0.4748 |
| Gemini - NER: gemini-3.1-flash-lite - MAP: gemini-3.5-flash | 50 | 55 | 9 | 0.4762 | 0.8475 | 0.6098 | 31 | 49 | 28 | 0.3875 | 0.5254 | 0.4460 |
| Gemini - NER: gemini-3-flash-preview - MAP: gemini-3.5-flash | 52 | 94 | 7 | 0.3562 | 0.8814 | 0.5073 | 32 | 60 | 27 | 0.3478 | 0.5424 | 0.4238 |


As can be seen the results improve significantly even when models not known for reasoning are used, so this indicates that the idea is correct and that this approach will bring better results as better models are used.

### Fifth test: LLM's test

#### NER stage

Since the model Gemini 3.5 Flash performed better no matter the test in the last tests then it will be used as the base case for testing the rest of the models. Now the models tested are, in addition to Gemini 3.5 Flash, the OpenAI models GPT-5.6 Terra, GPT-5.6 Luna, GPT-5.4 nano, GPT-5.4 Mini, GPT-5 Mini, GPT-4o Mini and GPT-4.1 Mini. At the same time the next Cluade models will be Claude Haiku 4.5 and Claude Sonnet 5.

Here are the results counting only NER F1:

| Test name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemini - gemini-3.5-flash | 50 | 54 | 9 | 0.4808 | 0.8475 | 0.6135 |
| Claude - claude-sonnet-5 | 35 | 42 | 24 | 0.4545 | 0.5932 | 0.5147 |
| Claude - claude-haiku-4-5-20251001 | 45 | 71 | 14 | 0.3879 | 0.7627 | 0.5143 |
| ChatGPT - gpt-4.1-mini | 39 | 58 | 20 | 0.4021 | 0.6610 | 0.5000 |
| ChatGPT - gpt-4o-mini | 32 | 38 | 27 | 0.4571 | 0.5424 | 0.4961 |
| ChatGPT - gpt-5.6-luna | 45 | 80 | 14 | 0.3600 | 0.7627 | 0.4891 |
| ChatGPT - gpt-5.4-mini | 47 | 93 | 12 | 0.3357 | 0.7966 | 0.4724 |
| ChatGPT - gpt-5.6-terra | 37 | 70 | 22 | 0.3458 | 0.6271 | 0.4458 |
| ChatGPT - gpt-5-mini | 41 | 86 | 18 | 0.3228 | 0.6949 | 0.4409 |
| ChatGPT - gpt-5.4-nano | 36 | 113 | 23 | 0.2416 | 0.6102 | 0.3462 |

Now, when analyzing the results from different models what was seen was that there was both a problem at measuring the F1 score and a normalization problem, which combined generated the loss of F1 score for the NER stage. In short, the evaluation metrics where made by comparing literally the texts and if a word was missing it was automatically seen as a false positive, therefore inflating the number. At the same time, phrases refering to the same phenotype that are written differently are considered two separate phenotypes, which is not correct. This last problem was also fixed by asking the model to use more standard or common medical terms.

This are the results after changing the prompts used and changing the evaluation are shown as follows:

```text
Eres un anotador clínico experto especializado en la Ontología de Fenotipos Humanos (HPO). Tu tarea es extraer signos, síntomas y anormalidades fenotípicas de la historia clínica.

REGLAS ESTRICTAS:
1. SOLO FENOTIPOS Y ANORMALIDADES: Extrae manifestaciones clínicas, signos físicos y síntomas reportados u observados.
2. ENFERMEDADES GLOBALES vs. FENOTIPOS ESPECÍFICOS: Ignora los diagnósticos de enfermedades sistémicas o síndromes globales (ej. "Lupus eritematoso sistémico", "Esclerosis sistémica"). Sin embargo, SÍ DEBES extraer anormalidades estructurales o inflamaciones específicas de órganos (ej. "pericarditis", "nefritis", "poliartritis").
3. MANEJO DE LABORATORIOS: No extraigas nombres de anticuerpos (ej. "ANA positivos", "anti-Scl70") ni valores numéricos crudos. SÍ puedes extraer alteraciones de laboratorio normalizadas que representen un fenotipo (ej. "proteinuria", "leucopenia").
4. NORMALIZACIÓN EXTREMA: El valor de "fenotipo" debe ser el concepto médico estandarizado más corto posible (idealmente 1 a 3 palabras). 
   - MAL: "rigidez matinal de más de una hora" -> BIEN: "rigidez matinal"
   - MAL: "proteinuria patológica (1.8 g/24h)" -> BIEN: "proteinuria"
   - MAL: "eritema fijo, plano, de bordes netos" -> BIEN: "eritema malar"
5. IGNORA todo síntoma negado ("sin fiebre") y antecedentes familiares.
6. Devuelve ÚNICAMENTE un objeto JSON válido, sin texto antes ni después.

EJEMPLO DE ENTRADA:
"Paciente con lupus. Presenta poliartritis simétrica y nefritis severa. Laboratorio: ANA positivo y proteinuria de 2g. Sin fiebre."

EJEMPLO DE SALIDA:
{
  "fenotipos": [
    {"fenotipo": "poliartritis", "contexto": "Presenta poliartritis simétrica y nefritis severa."},
    {"fenotipo": "nefritis", "contexto": "Presenta poliartritis simétrica y nefritis severa."},
    {"fenotipo": "proteinuria", "contexto": "Laboratorio: ANA positivo y proteinuria de 2g."}
  ]
}
```

| Test name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ChatGPT - gpt-4o-mini | 48 | 15 | 11 | 0.7619 | 0.8136 | 0.7869 |
| Claude - claude-haiku-4-5-20251001 | 48 | 21 | 11 | 0.6957 | 0.8136 | 0.7500 |
| Claude - claude-sonnet-5 | 52 | 30 | 7 | 0.6341 | 0.8814 | 0.7376 |
| ChatGPT - gpt-4.1-mini | 46 | 28 | 13 | 0.6216 | 0.7797 | 0.6917 |
| ChatGPT - gpt-5.6-terra | 47 | 32 | 12 | 0.5949 | 0.7966 | 0.6812 |
| ChatGPT - gpt-5.6-luna | 46 | 33 | 13 | 0.5823 | 0.7797 | 0.6667 |
| Gemini - gemini-3.5-flash | 48 | 38 | 11 | 0.5581 | 0.8136 | 0.6621 |
| ChatGPT - gpt-5.4-mini | 47 | 36 | 12 | 0.5663 | 0.7966 | 0.6620 |
| ChatGPT - gpt-5-mini | 47 | 44 | 12 | 0.5165 | 0.7966 | 0.6267 |
| ChatGPT - gpt-5.4-nano | 42 | 35 | 17 | 0.5455 | 0.7119 | 0.6176 |

The results show a significant improvement at the cost of specificity. Therefore the phenotypes are detected but at the cost of specificity. Another test was done with another prompt and the results are shown below:

```text
Eres un anotador clínico experto especializado en la Ontología de Fenotipos Humanos (HPO). Tu tarea es extraer signos, síntomas y anormalidades fenotípicas de la historia clínica.

REGLAS ESTRICTAS DE EXTRACCIÓN:
1. EXTRACCIÓN ESPECÍFICA (GRANULARIDAD): Extrae el síntoma manteniendo sus modificadores clínicos clave (tipo, anatomía, lateralidad, severidad). Usa las palabras literales del texto siempre que sea posible.
   - BIEN: "disfagia motora", "poliartritis simétrica", "dolor abdominal severo", "eritema malar clásico".
2. ELIMINA TIEMPOS Y EXPLICACIONES: El fenotipo NO debe contener descripciones de duración, frases conectoras ni explicaciones.
   - MAL: "rigidez matinal de más de una hora" -> BIEN: "rigidez matinal"
   - MAL: "pérdida de cabello difusa que el especialista diagnostica como alopecia" -> BIEN: "alopecia no cicatricial" o "pérdida de cabello difusa"
3. ENFERMEDADES GLOBALES vs. FENOTIPOS: Ignora diagnósticos de enfermedades sistémicas (ej. "Lupus", "Espondilitis"). SÍ extrae manifestaciones específicas de órganos (ej. "pericarditis", "nefritis lúpica", "sacroiliítis bilateral").
4. LABORATORIOS: Ignora anticuerpos (ej. "ANA positivos") y marcadores inflamatorios ("PCR elevada"). SÍ extrae anormalidades fisiológicas base (ej. "proteinuria", "anemia megaloblástica").
5. IGNORA síntomas negados ("sin fiebre") y antecedentes familiares.
6. Devuelve ÚNICAMENTE un objeto JSON válido, sin texto adicional.

EJEMPLO DE ENTRADA:
"Paciente con lupus. Presenta poliartritis simétrica severa desde hace 2 meses y nefritis. Laboratorio: ANA positivo y proteinuria de 2g. Sin fiebre."

EJEMPLO DE SALIDA:
{
  "fenotipos": [
    {"fenotipo": "poliartritis simétrica severa", "contexto": "Presenta poliartritis simétrica severa desde hace 2 meses y nefritis."},
    {"fenotipo": "nefritis", "contexto": "Presenta poliartritis simétrica severa desde hace 2 meses y nefritis."},
    {"fenotipo": "proteinuria", "contexto": "Laboratorio: ANA positivo y proteinuria de 2g."}
  ]
}
```

| Test name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ChatGPT - gpt-4o-mini | 44 | 23 | 15 | 0.6567 | 0.7458 | 0.6984 |
| ChatGPT - gpt-5.4-mini | 45 | 34 | 14 | 0.5696 | 0.7627 | 0.6522 |
| Claude - claude-haiku-4-5-20251001 | 42 | 38 | 17 | 0.5250 | 0.7119 | 0.6043 |
| Gemini - gemini-3.5-flash | 40 | 43 | 19 | 0.4819 | 0.6780 | 0.5634 |
| ChatGPT - gpt-4.1-mini | 38 | 38 | 21 | 0.5000 | 0.6441 | 0.5630 |
| Claude - claude-sonnet-5 | 39 | 43 | 20 | 0.4756 | 0.6610 | 0.5532 |
| ChatGPT - gpt-5.4-nano | 41 | 59 | 18 | 0.4100 | 0.6949 | 0.5157 |
| ChatGPT - gpt-5.6-luna | 36 | 51 | 23 | 0.4138 | 0.6102 | 0.4932 |
| ChatGPT - gpt-5.6-terra | 32 | 55 | 27 | 0.3678 | 0.5424 | 0.4384 |
| ChatGPT - gpt-5-mini | 36 | 72 | 23 | 0.3333 | 0.6102 | 0.4311 |

This prompt was evaluated in strict text comparisons and as it can be seen the results are still better than in the previous section of tests but, more importantly, they have more specificity than the previous test. What is clear, however, is that the model GPT 4o Mini is the one with better performance despite being more modern examples like GPT Luna and Terra available.

#### Mapping stage

Now more models will be used to reason in the RAG stage in order to see which are the best ones for this stage. From now on the tests will be made using strict strings comparisons in order to be able to compare the results and because it makes no sense to change the way it is measured to a less precise one.

For the first prompt tested (the one with more general phenotypes) the results are as follows:

| Model name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 score | Map TP | Map FP | Map FN | Map Prec | Map Rec | Final F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.6-luna) | 29 | 30 | 30 | 0.4915 | 0.4915 | 0.4915 | 32 | 23 | 27 | 0.5818 | 0.5424 | 0.5614 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-4o-mini) | 32 | 29 | 27 | 0.5246 | 0.5424 | 0.5333 | 31 | 23 | 28 | 0.5741 | 0.5254 | 0.5487 |
| NER: gpt-4o-mini - MAP: Gemini (gemini-3.5-flash) | 30 | 34 | 29 | 0.4688 | 0.5085 | 0.4878 | 30 | 24 | 29 | 0.5556 | 0.5085 | 0.5310 |
| NER: gpt-4o-mini - MAP: Gemini (gemini-3.1-flash-lite) | 30 | 27 | 29 | 0.5263 | 0.5085 | 0.5172 | 29 | 22 | 30 | 0.5686 | 0.4915 | 0.5273 |
| NER: gpt-4o-mini - MAP: Claude (claude-sonnet-5) | 33 | 31 | 26 | 0.5156 | 0.5593 | 0.5366 | 30 | 26 | 29 | 0.5357 | 0.5085 | 0.5217 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5-mini) | 30 | 29 | 29 | 0.5085 | 0.5085 | 0.5085 | 29 | 26 | 30 | 0.5273 | 0.4915 | 0.5088 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.6-terra) | 32 | 33 | 27 | 0.4923 | 0.5424 | 0.5161 | 29 | 26 | 30 | 0.5273 | 0.4915 | 0.5088 |
| NER: gpt-4o-mini - MAP: Gemini (gemini-3-flash-preview) | 29 | 35 | 30 | 0.4531 | 0.4915 | 0.4715 | 29 | 26 | 30 | 0.5273 | 0.4915 | 0.5088 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.4-mini) | 31 | 36 | 28 | 0.4627 | 0.5254 | 0.4921 | 29 | 30 | 30 | 0.4915 | 0.4915 | 0.4915 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.4-nano) | 28 | 35 | 31 | 0.4444 | 0.4746 | 0.4590 | 28 | 29 | 31 | 0.4912 | 0.4746 | 0.4828 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5-2025-08-07) | 31 | 35 | 28 | 0.4697 | 0.5254 | 0.4960 | 28 | 30 | 31 | 0.4828 | 0.4746 | 0.4786 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.6-sol) | 31 | 32 | 28 | 0.4921 | 0.5254 | 0.5082 | 27 | 28 | 32 | 0.4909 | 0.4576 | 0.4737 |
| NER: gpt-4o-mini - MAP: Claude (claude-haiku-4-5-20251001) | 30 | 31 | 29 | 0.4918 | 0.5085 | 0.5000 | 27 | 29 | 32 | 0.4821 | 0.4576 | 0.4696 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-4.1-mini) | 29 | 30 | 30 | 0.4915 | 0.4915 | 0.4915 | 26 | 27 | 33 | 0.4906 | 0.4407 | 0.4643 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.2-2025-12-11) | 30 | 36 | 29 | 0.4545 | 0.5085 | 0.4800 | 26 | 33 | 33 | 0.4407 | 0.4407 | 0.4407 |

For the second prompt tested (the one with stricter phenotypes) the results are as follows:

| Model name | NER TP | NER FP | NER FN | NER Prec | NER Rec | NER F1 score | Map TP | Map FP | Map FN | Map Prec | Map Rec | Final F1 score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NER: gpt-4o-mini - MAP: Gemini (gemini-3.5-flash) | 44 | 23 | 15 | 0.6567 | 0.7458 | 0.6984 | 27 | 30 | 32 | 0.4737 | 0.4576 | 0.4655 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.6-sol) | 44 | 28 | 15 | 0.6111 | 0.7458 | 0.6718 | 27 | 33 | 32 | 0.4500 | 0.4576 | 0.4538 |
| NER: gpt-4o-mini - MAP: Gemini (gemini-3.1-flash-lite) | 43 | 23 | 16 | 0.6515 | 0.7288 | 0.6880 | 26 | 30 | 33 | 0.4643 | 0.4407 | 0.4522 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.4-nano) | 45 | 29 | 14 | 0.6081 | 0.7627 | 0.6767 | 28 | 37 | 31 | 0.4308 | 0.4746 | 0.4516 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.6-terra) | 46 | 29 | 13 | 0.6133 | 0.7797 | 0.6866 | 27 | 35 | 32 | 0.4355 | 0.4576 | 0.4463 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-4o-mini) | 45 | 26 | 14 | 0.6338 | 0.7627 | 0.6923 | 27 | 35 | 32 | 0.4355 | 0.4576 | 0.4463 |
| NER: gpt-4o-mini - MAP: Claude (claude-sonnet-5) | 41 | 30 | 18 | 0.5775 | 0.6949 | 0.6308 | 26 | 32 | 33 | 0.4483 | 0.4407 | 0.4444 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5-2025-08-07) | 44 | 27 | 15 | 0.6197 | 0.7458 | 0.6769 | 26 | 33 | 33 | 0.4407 | 0.4407 | 0.4407 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.6-luna) | 42 | 31 | 17 | 0.5753 | 0.7119 | 0.6364 | 26 | 35 | 33 | 0.4262 | 0.4407 | 0.4333 |
| NER: gpt-4o-mini - MAP: Gemini (gemini-3-flash-preview) | 43 | 29 | 16 | 0.5972 | 0.7288 | 0.6565 | 25 | 34 | 34 | 0.4237 | 0.4237 | 0.4237 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.4-mini) | 43 | 26 | 16 | 0.6232 | 0.7288 | 0.6719 | 25 | 34 | 34 | 0.4237 | 0.4237 | 0.4237 |
| NER: gpt-4o-mini - MAP: Claude (claude-haiku-4-5-20251001) | 45 | 31 | 14 | 0.5921 | 0.7627 | 0.6667 | 25 | 38 | 34 | 0.3968 | 0.4237 | 0.4098 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5.2-2025-12-11) | 46 | 24 | 13 | 0.6571 | 0.7797 | 0.7132 | 24 | 37 | 35 | 0.3934 | 0.4068 | 0.4000 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-5-mini) | 41 | 30 | 18 | 0.5775 | 0.6949 | 0.6308 | 24 | 37 | 35 | 0.3934 | 0.4068 | 0.4000 |
| NER: gpt-4o-mini - MAP: OpenAI (gpt-4.1-mini) | 41 | 32 | 18 | 0.5616 | 0.6949 | 0.6212 | 22 | 40 | 37 | 0.3548 | 0.3729 | 0.3636 |

Surprisingly the prompt with more general phenotypes performs better End-To-End, but worse in the NER stage and viceversa. By looking at the results of the tests, however, it was seen that some reasonings were cut short and some others did not have the possible HPO code in their possibilities, so another test was made using both scripts using a larger context and a larger Top-k list of 10:

| NER: gpt-4o-mini gral-ner-prompt - MAP: Claude (claude-opus-5) | 29 | 33 | 30 | 0.4677 | 0.4915 | 0.4793 | 31 | 26 | 28 | 0.5439 | 0.5254 | 0.5345 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5.6-terra) | 29 | 37 | 30 | 0.4394 | 0.4915 | 0.4640 | 31 | 31 | 28 | 0.5000 | 0.5254 | 0.5124 |
| NER: gpt-4o-mini gral-ner-prompt - MAP: Claude (claude-sonnet-5) | 29 | 30 | 30 | 0.4915 | 0.4915 | 0.4915 | 29 | 26 | 30 | 0.5273 | 0.4915 | 0.5088 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: Gemini (gemini-3.5-flash) | 32 | 33 | 27 | 0.4923 | 0.5424 | 0.5161 | 29 | 27 | 30 | 0.5179 | 0.4915 | 0.5043 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5.4-mini) | 30 | 31 | 29 | 0.4918 | 0.5085 | 0.5000 | 30 | 30 | 29 | 0.5000 | 0.5085 | 0.5042 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5-2025-08-07) | 30 | 31 | 29 | 0.4918 | 0.5085 | 0.5000 | 29 | 29 | 30 | 0.5000 | 0.4915 | 0.4957 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: Gemini (gemini-3.1-flash-lite) | 30 | 31 | 29 | 0.4918 | 0.5085 | 0.5000 | 28 | 28 | 31 | 0.5000 | 0.4746 | 0.4870 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5.6-luna) | 31 | 37 | 28 | 0.4559 | 0.5254 | 0.4882 | 30 | 37 | 29 | 0.4478 | 0.5085 | 0.4762 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5-mini) | 28 | 34 | 31 | 0.4516 | 0.4746 | 0.4628 | 28 | 33 | 31 | 0.4590 | 0.4746 | 0.4667 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-4o-mini) | 29 | 32 | 30 | 0.4754 | 0.4915 | 0.4833 | 27 | 31 | 32 | 0.4655 | 0.4576 | 0.4615 |
| NER: gpt-4o-mini specific-ner-prompt - MAP: Claude (claude-opus-5) | 45 | 30 | 14 | 0.6000 | 0.7627 | 0.6716 | 29 | 38 | 30 | 0.4328 | 0.4915 | 0.4603 |
| NER: gpt-4o-mini specific-ner-prompt - MAP: Claude (claude-sonnet-5) | 49 | 24 | 10 | 0.6712 | 0.8305 | 0.7424 | 28 | 35 | 31 | 0.4444 | 0.4746 | 0.4590 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5.4-nano) | 32 | 32 | 27 | 0.5000 | 0.5424 | 0.5203 | 28 | 36 | 31 | 0.4375 | 0.4746 | 0.4553 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: OpenAI (gpt-5.6-sol) | 30 | 33 | 29 | 0.4762 | 0.5085 | 0.4918 | 27 | 33 | 32 | 0.4500 | 0.4576 | 0.4538 |
| NER: gpt-4o-mini (gral-ner-prompt) - MAP: Gemini (gemini-3-flash-preview) | 30 | 34 | 29 | 0.4688 | 0.5085 | 0.4878 | 26 | 31 | 33 | 0.4561 | 0.4407 | 0.4483 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5.6-sol) | 47 | 25 | 12 | 0.6528 | 0.7966 | 0.7176 | 28 | 40 | 31 | 0.4118 | 0.4746 | 0.4409 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5.6-terra) | 42 | 30 | 17 | 0.5833 | 0.7119 | 0.6412 | 27 | 38 | 32 | 0.4154 | 0.4576 | 0.4355 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: Gemini (gemini-3.1-flash-lite) | 44 | 30 | 15 | 0.5946 | 0.7458 | 0.6617 | 26 | 36 | 33 | 0.4194 | 0.4407 | 0.4298 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5.4-mini) | 45 | 24 | 14 | 0.6522 | 0.7627 | 0.7031 | 26 | 39 | 33 | 0.4000 | 0.4407 | 0.4194 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: Gemini (gemini-3-flash-preview) | 45 | 24 | 14 | 0.6522 | 0.7627 | 0.7031 | 24 | 34 | 35 | 0.4138 | 0.4068 | 0.4103 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-4o-mini) | 46 | 29 | 13 | 0.6133 | 0.7797 | 0.6866 | 26 | 46 | 33 | 0.3611 | 0.4407 | 0.3969 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5.4-nano) | 44 | 25 | 15 | 0.6377 | 0.7458 | 0.6875 | 24 | 41 | 35 | 0.3692 | 0.4068 | 0.3871 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5.6-luna) | 43 | 27 | 16 | 0.6143 | 0.7288 | 0.6667 | 24 | 41 | 35 | 0.3692 | 0.4068 | 0.3871 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5-mini) | 44 | 29 | 15 | 0.6027 | 0.7458 | 0.6667 | 24 | 42 | 35 | 0.3636 | 0.4068 | 0.3840 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: OpenAI (gpt-5-2025-08-07) | 48 | 23 | 11 | 0.6761 | 0.8136 | 0.7385 | 24 | 42 | 35 | 0.3636 | 0.4068 | 0.3840 |
| NER: gpt-4o-mini (specific-ner-prompt) - MAP: Gemini (gemini-3.5-flash) | 39 | 32 | 20 | 0.5493 | 0.6610 | 0.6000 | 23 | 38 | 36 | 0.3770 | 0.3898 | 0.3833 |


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