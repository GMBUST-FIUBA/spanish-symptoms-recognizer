# Sypmtoms recognizer in spanish

## Introduction
This work is made as part of a final project in the University of Buenos Aires, Argentina, named *Detection of Genetic and Rare Deseases* (*Detección de enfermedades genéticas poco frecuentes*, in spanish).

Based on the model developed by the Barcelona Supercomputing Center (BSC) named [bsc-bio-ehr-es](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es), a fine-tuned version of said model is made in order to recognize symptoms in clynical history. Those recognized symptoms can then be mapped on to an ontology, however nowadays the only ontology used is [HPO](https://hpo.jax.org/).

## Training
The model was trained using the following datasets:

- [SympTEMIST Corpus](https://zenodo.org/records/10635215)
- IN PROGRESS: [Chilean Waiting List](https://zenodo.org/records/7555181)
- IN THE FUTURE: [eHealth-KD v2](https://zenodo.org/records/3696792)
- IN THE FUTURE: [NUBes](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus)


## Mapping
As stated before, the only ontology that it it supported is [HPO](https://hpo.jax.org/). The translations used are those made by the 
Human Phenotype Ontology Internationalisation Effort's [spanish translation](https://github.com/obophenotype/hpo-translations/blob/main/babelon/hp-es.babelon.tsv).