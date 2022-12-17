#!/bin/bash

set -e

INPUT_FILE="${1:-/dev/stdin}"

MODEL_DIR="${MODEL_DIR:-model}"

RLEV="python rlev.py"
TITLE_VECTORIZER="$DATA_DIR/title-vectorizer.pickle"
ABSTR_VECTORIZER="$DATA_DIR/abstr-vectorizer.pickle"
WORD_FEATURE_MODEL="$DATA_DIR/word-feature-model.pickle"
RLEV_PRIORS="$DATA_DIR/rlev-priors.pickle"
COMBINED_MODEL="$DATA_DIR/combined-model.pickle"

$RLEV get-combined-model-predictions "$INPUT_FILE" \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER" \
    --word-feature-model "$WORD_FEATURE_MODEL" \
    --rlev-priors "$RLEV_PRIORS" \
    --combined-model "$COMBINED_MODEL"
