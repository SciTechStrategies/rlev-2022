# RLEV 2022

This repository contains the code for building a research-level prediction model.

The process of building the model is described in this paper:
https://www.sciencedirect.com/science/article/abs/pii/S1751157713000825

## Installation

The code is written in Python. We ran the model using Python version 3.10.4.
We include an `environment.yml` file for building a [Conda] environment with the correct Python version.

Dependencies can be installed via [pip]: `pip install -r requirements.txt`.

## Testing

A small test suite is included to ensure that a model can be built and tested.
You can run this end-to-end test via: `./integration-testing/integration-test.sh`.
That will build a small model based on some test input data, and then generate predictions and print them to stdout.

# Building models

We include a `train.sh` script to simplify model building.
The script expects a single input file as an argument, or you can pipe stdin into the script.
The input file should be tab-delimited with eight columns:

* Paper ID
* Research level (an integer between 1-4)
* Ratio of cited papers belonging to researh level 1 (a floating-point value between 0 and 1)
* Ratio of cited papers belonging to researh level 2 (a floating-point value between 0 and 1)
* Ratio of cited papers belonging to researh level 3 (a floating-point value between 0 and 1)
* Ratio of cited papers belonging to researh level 4 (a floating-point value between 0 and 1)
* Title
* Abstract

You can set three environment variables that will affect the output:

* `MODEL_DIR` the directory where model files should be put (defaults to `model/`)
* `DATA_DIR` the directory where intermediate data files should be put (defaults to `data/`)
* `MIN_WORD_FEATURE_DF` the minimum number of documents (titles or abstracts) that a word must appear in to be included as a feature.

Output of this command will be a set of files that contain [pickled] Python objects that are used to perform the prediction task.

# Predicting

You can use a model to predict research level for a set of papers.

We include a `predict.sh` script that can be used to get predictions from a model.
The script can read a filename or you can pipe stdin into the script.
The input of the predict script should be tab-delimited text with seven columns.
The columns are the same as the eight columns used for building the model, with the second column (research level) removed.

The model that we built is available for download.
