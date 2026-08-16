# Data generation

## Introduction

Following the schema from the BSC for the fine-tuning of their model, we make functions for the data we want to create and have the synthetic data and organic data in different folders.

## Execution

To execute the data generation, use this command:

`python3 -m symptoms_recognizer.ner_model.data.utils.datasets_generator`

This will create the data in the root folder.

If you want to store the data elsewhere you have to add the `--output_folder argument` like this:

`python3 -m symptoms_recognizer.ner_model.data.utils.datasets_generator --output_folder symptoms_recognizer/ner_model/data`