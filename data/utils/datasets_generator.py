import argparse
import csv
import os

import symptemist.symptemist_brat_to_bio_conv

TRAINING_DATASETS_PATHS_FILE = "utils/datasets_train_paths.csv"
TESTING_DATASETS_PATHS_FILE = "utils/datasets_test_paths.csv"

TRAINING_DATASET_OUTPUT_NAME = "train_set.jsonl"

TRAINING_DATASET_CONVERTERS = {
    "SYMPTEMIST" : symptemist.symptemist_brat_to_bio_conv.convert_brat_to_bio
}

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_set",
                        default=0.3,
                        help="Number between 0 and 1 to take a part from training set and use it as a validation set.")
    parser.add_argument("--output_folder", default=None, help="Output folder for JSONL files")
    parser.add_argument("--max_length", default=512, help="Maximum of tokens per sentence")

    args = parser.parse_args()

    # Open output file
    output_file_path = TRAINING_DATASET_OUTPUT_NAME

    if args.output_folder is not None:
        output_file_path = os.path.join(args.output_folder, output_file_path)

    with open(output_file_path, "w") as output_train_set:

        # Open training datasets file
        with open(TRAINING_DATASETS_PATHS_FILE, mode="r", encoding="utf-8") as training_sets_paths_file:

            # Get training datsets paths
            training_datasets_paths = csv.reader(training_sets_paths_file)
            for line in training_datasets_paths:
                # Get data from line
                print(line)
                name = line[0]
                path = line[1]

                # Use converter
                TRAINING_DATASET_CONVERTERS[name](path, output_train_set)