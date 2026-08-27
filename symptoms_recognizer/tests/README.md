# Testing

## Introduction

This folder contains an End-to-End (E2E) evaluation pipeline designed to assess the performance of the phenotype recognizer. Models are benchmarked primarily using the micro/macro F1 score alongside complementary metrics, comparing different inference strategies and architectural approaches.

The models tested in this benchmark include:

- [es-hpo-phenotype-detector](https://huggingface.co/GonzaloMB/es-hpo-phenotype-detector): Trained internally using the scripts in `symptoms_recognizer/ner_model/training`.
- [HUMADEX/spanish_medical_ner](https://huggingface.co/HUMADEX/spanish_medical_ner)
- [medspaner/roberta-es-clinical-trials-umls-7sgs-ner](https://huggingface.co/medspaner/roberta-es-clinical-trials-umls-7sgs-ner)

The idea is not only to find the right combination of parameters for the clinical records analysis, but also to find the best way to parse the records to extract the most information and avoid the most noise possible. Several strategies where tested: sentence by sentence analysis, chunks analysis and themed selection texts sections analysis. All of those strategies are explained in the next sections.

## Types of text parsing

## Results

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