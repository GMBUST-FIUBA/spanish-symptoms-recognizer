import argparse
import csv
import os

import symptemist.symptemist_brat_to_bio_conv

TRAINING_DATASETS_PATHS_FILE = "utils/datasets_train_paths.csv"
TESTING_DATASETS_PATHS_FILE = "utils/datasets_test_paths.csv"

TRAINING_DATASET_OUTPUT_NAME = "train_set.jsonl"
TESTING_DATASET_OUTPUT_NAME = "test_set.jsonl"

TRAINING_DATASET_CONVERTERS = {
    "SYMPTEMIST" : symptemist.symptemist_brat_to_bio_conv.convert_brat_to_bio
}

TESTING_DATASET_CONVERTERS = {
    "SYMPTEMIST" : symptemist.symptemist_brat_to_bio_conv.convert_brat_to_bio
}

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_train_set", action="store_true", help="Write if train set is wanted")
    parser.add_argument("--generate_test_set", action="store_true", help="Write if test set is wanted")
    parser.add_argument("--generate_validation_set",
                        default=None,
                        help="Number between 0 and 1 to " \
                        "take a part from training set and use it as a validation set. " \
                        "Don't add if not wanted")
    parser.add_argument("--output_folder", default=None, help="Output folder for JSONL files")
    parser.add_argument("--max_length", default=512, help="Maximum of tokens per sentence")

    args = parser.parse_args()

    # Open output training file
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
                name = line[0]
                path = line[1]

                # Use converter
                TRAINING_DATASET_CONVERTERS[name](path, output_train_set)

    # Open output test file
    output_file_path = TESTING_DATASET_OUTPUT_NAME

    if args.output_folder is not None:
        output_file_path = os.path.join(args.output_folder, output_file_path)

    with open(output_file_path, "w") as output_test_set:

        # Open testing datasets file
        with open(TESTING_DATASETS_PATHS_FILE, mode="r", encoding="utf-8") as testing_sets_paths_file:

            # Get testing datsets paths
            testing_datasets_paths = csv.reader(testing_sets_paths_file)
            for line in testing_datasets_paths:
                # Get data from line
                name = line[0]
                path = line[1]

                # Use converter
                TESTING_DATASET_CONVERTERS[name](path, output_test_set)