#!/bin/bash

set -e

RLEV="python rlev.py"
DATA_DIR="data"
MIN_WORD_FEATURE_DF=250
MIN_YR=10
MAX_YR=13

TITLE_COL=8
ABSTR_COL=9
RLEV_COL=2

mkdir -p $DATA_DIR

echo "Downloading data from S3..."
ALL_INPUT_FILES=""
for yr in $(seq $MIN_YR $MAX_YR)
do
    echo "Downloading $yr..."
    basename="cpid${yr}_rl_nr_fx_t_a.txt.gz"
    # aws s3 cp "s3://scitech/projects/rlev/$basename" $DATA_DIR
    ALL_INPUT_FILES="$ALL_INPUT_FILES $DATA_DIR/$basename"
done

UNZIP_INPUT="gunzip -c $ALL_INPUT_FILES"  #  | head -1000000"

echo "Building title vectorizer..."
TITLE_VECTORIZER="$DATA_DIR/title-vectorizer.pickle"
eval $UNZIP_INPUT | \
    awk 'BEGIN{FS="\t"}{print $'$TITLE_COL'}' | \
    $RLEV create-count-vectorizer - "$TITLE_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building abstract vectorizer..."
ABSTR_VECTORIZER="$DATA_DIR/abstr-vectorizer.pickle"
eval $UNZIP_INPUT | \
    awk 'BEGIN{FS="\t"}{print $'$ABSTR_COL'}' | \
    $RLEV create-count-vectorizer - "$ABSTR_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building word feature model inputs..."
WORD_FEATURE_INPUTS="$DATA_DIR/word-feature-model-inputs.pickle"
eval $UNZIP_INPUT | \
    awk 'BEGIN{FS="\t";OFS="\t"}{print $'$RLEV_COL',$'$TITLE_COL',$'$ABSTR_COL'}' | \
    $RLEV create-word-feature-model-inputs - "$WORD_FEATURE_INPUTS" \
    --title-vectorizer "$TITLE_VECTORIZER" \
    --abstr-vectorizer "$ABSTR_VECTORIZER"

echo "Training model..."
WORD_FEATURE_MODEL="$DATA_DIR/word-feature-model.pickle"
$RLEV train-word-feature-model "$WORD_FEATURE_INPUTS" "$WORD_FEATURE_MODEL"
