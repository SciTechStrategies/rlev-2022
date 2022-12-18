#!/bin/bash

set -e

DATA_DIR=integration-testing/data
export DATA_DIR
MODEL_DIR=integration-testing/model
export MODEL_DIR
MIN_WORD_FEATURE_DF=25
export MIN_WORD_FEATURE_DF

INPUT_FILE=integration-testing/cpid_rl_fx_t_a.txt.gz

gunzip -c "$INPUT_FILE" | ./train.sh

gunzip -c "$INPUT_FILE" | \
    awk 'BEGIN{FS=OFS="\t"}{print $1,$3,$4,$5,$6,$7,$8,$9}' | \
    ./predict.sh
