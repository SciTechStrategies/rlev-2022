#!/bin/bash

set -e

RLEV="python rlev.py"
DATA_DIR="data"
MIN_WORD_FEATURE_DF=250
MIN_YR=10
MAX_YR=21

RLEV_COL=2
TITLE_COL=7
ABSTR_COL=8

mkdir -p $DATA_DIR

echo "Downloading data from S3..."
ALL_INPUT_FILES=""
for yr in $(seq $MIN_YR $MAX_YR)
do
    echo "Downloading 20$yr..."
    basename="cpid${yr}_rl_nr_fx_t_a.txt.gz"
    aws s3 cp "s3://scitech/projects/rlev/$basename" $DATA_DIR
    ALL_INPUT_FILES="$ALL_INPUT_FILES $DATA_DIR/$basename"
done

echo "Formatting data..."
FORMATTED="$DATA_DIR/cpid_rl_fx_t_a_${MIN_YR}_${MAX_YR}.txt"
gunzip -c $ALL_INPUT_FILES | $RLEV format-input - > $FORMATTED

echo "Building title vectorizer..."
TITLE_VECTORIZER="$DATA_DIR/title-vectorizer.pickle"
awk 'BEGIN{FS="\t"}{print $'$TITLE_COL'}' "$FORMATTED" | \
    $RLEV create-count-vectorizer - "$TITLE_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building abstract vectorizer..."
ABSTR_VECTORIZER="$DATA_DIR/abstr-vectorizer.pickle"
awk 'BEGIN{FS="\t"}{print $'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV create-count-vectorizer - "$ABSTR_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building word feature model inputs..."
WORD_FEATURE_INPUTS="$DATA_DIR/word-feature-model-inputs.pickle"
awk 'BEGIN{FS="\t";OFS="\t"}{print $'$RLEV_COL',$'$TITLE_COL',$'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV create-word-feature-model-inputs - "$WORD_FEATURE_INPUTS" \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER"

echo "Training word feature model..."
WORD_FEATURE_MODEL="$DATA_DIR/word-feature-model.pickle"
$RLEV train-lr-model "$WORD_FEATURE_INPUTS" "$WORD_FEATURE_MODEL"

echo "Getting rlev priors..."
RLEV_PRIORS="$DATA_DIR/rlev-priors.pickle"
awk 'BEGIN{FS="\t"}{print $'$RLEV_COL'}' "$FORMATTED" | \
    $RLEV get-rlev-priors - "$RLEV_PRIORS"

echo "Building combined model inputs..."
COMBINED_INPUTS="$DATA_DIR/combined-model-inputs.pickle"
awk 'BEGIN{FS="\t";OFS="\t"}{print $'$RLEV_COL',$3,$4,$5,$6,$'$TITLE_COL',$'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV create-combined-model-inputs - "$COMBINED_INPUTS" \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER" \
    --word-feature-model "$WORD_FEATURE_MODEL" \
    --rlev-priors "$RLEV_PRIORS"

echo "Training combined model..."
COMBINED_MODEL="$DATA_DIR/combined-model.pickle"
$RLEV train-lr-model "$COMBINED_INPUTS" "$COMBINED_MODEL"

echo "Getting combined model predictions..."
PREDICTIONS="$DATA_DIR/predictions.txt"
awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$'$RLEV_COL',$3,$4,$5,$6,$'$TITLE_COL',$'$ABSTR_COL'}' "$FORMATTED" | \
    $RLEV get-combined-model-predictions - \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER" \
    --word-feature-model "$WORD_FEATURE_MODEL" \
    --rlev-priors "$RLEV_PRIORS" \
    --combined-model "$COMBINED_MODEL" > "$PREDICTIONS"

gzip "$PREDICTIONS"

aws s3 cp "$PREDICTIONS".gz s3://scitech/projects/rlev/
aws s3 cp "$COMBINED_MODEL" s3://scitech/projects/rlev/
aws s3 cp "$COMBINED_INPUTS" s3://scitech/projects/rlev/
aws s3 cp "$RLEV_PRIORS" s3://scitech/projects/rlev/
aws s3 cp "$WORD_FEATURE_MODEL" s3://scitech/projects/rlev/
aws s3 cp "$WORD_FEATURE_INPUTS" s3://scitech/projects/rlev/
aws s3 cp "$ABSTR_VECTORIZER" s3://scitech/projects/rlev/
aws s3 cp "$TITLE_VECTORIZER" s3://scitech/projects/rlev/
