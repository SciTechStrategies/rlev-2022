#!/bin/bash

set -e

INPUT_FILE="${1:-/dev/stdin}"

MODEL_DIR="${MODEL_DIR:-model}"
DATA_DIR="${DATA_DIR:-data}"

mkdir -p "$MODEL_DIR"
mkdir -p "$DATA_DIR"

MIN_WORD_FEATURE_DF="${MIN_WORD_FEATURE_DF:-250}"

RLEV_COL=2
TITLE_COL=7
ABSTR_COL=8

RLEV="python rlev.py"

echo "Formatting data..."
FORMATTED="$DATA_DIR/cpid_rl_fx_t_a.txt"
$RLEV format-input --with-labels "$INPUT_FILE" > "$FORMATTED"

echo "Building title vectorizer..."
TITLE_VECTORIZER="$MODEL_DIR/title-vectorizer.pickle"
awk 'BEGIN{FS="\t"}{print $'$TITLE_COL'}' "$FORMATTED" | \
    $RLEV create-count-vectorizer - "$TITLE_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building abstract vectorizer..."
ABSTR_VECTORIZER="$MODEL_DIR/abstr-vectorizer.pickle"
awk 'BEGIN{FS="\t"}{print $'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV create-count-vectorizer - "$ABSTR_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building word feature model inputs..."
WORD_FEATURE_INPUTS="$MODEL_DIR/word-feature-model-inputs.pickle"
awk 'BEGIN{FS="\t";OFS="\t"}{print $'$RLEV_COL',$'$TITLE_COL',$'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV create-word-feature-model-inputs - "$WORD_FEATURE_INPUTS" \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER"

echo "Training word feature model..."
WORD_FEATURE_MODEL="$MODEL_DIR/word-feature-model.pickle"
$RLEV train-lr-model "$WORD_FEATURE_INPUTS" "$WORD_FEATURE_MODEL"

echo "Getting rlev priors..."
RLEV_PRIORS="$MODEL_DIR/rlev-priors.pickle"
awk 'BEGIN{FS="\t"}{print $'$RLEV_COL'}' "$FORMATTED" | \
    $RLEV get-rlev-priors - "$RLEV_PRIORS"

echo "Building combined model inputs..."
COMBINED_INPUTS="$MODEL_DIR/combined-model-inputs.pickle"
awk 'BEGIN{FS="\t";OFS="\t"}{print $'$RLEV_COL',$3,$4,$5,$6,$'$TITLE_COL',$'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV create-combined-model-inputs - "$COMBINED_INPUTS" \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER" \
    --word-feature-model "$WORD_FEATURE_MODEL" \
    --rlev-priors "$RLEV_PRIORS"

echo "Training combined model..."
COMBINED_MODEL="$MODEL_DIR/combined-model.pickle"
$RLEV train-lr-model "$COMBINED_INPUTS" "$COMBINED_MODEL"
