#!/bin/bash

RLEV="python rlev.py"
DATA_DIR="data"
MIN_WORD_FEATURE_DF=10
MIN_YR=10
MAX_YR=10

mkdir -p $DATA_DIR

echo "Downloading data from S3..."
for yr in $(seq $MIN_YR $MAX_YR)
do
    echo $yr
    # aws s3 cp "s3://scitech/projects/rlev/cpid${yr}_rl_nr_fx_t_a.txt.gz" $DATA_DIR
done

echo "Building title vectorizer..."
TITLE_VECTORIZER="$DATA_DIR/title-vectorizer.pickle"
gunzip -c $DATA_DIR/cpid*_rl_nr_fx_t_a.txt.gz | \
    awk 'BEGIN{FS="\t"}{print $8}' | \
    head -10000 | \
    $RLEV create-count-vectorizer - "$TITLE_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"

echo "Building abstract vectorizer..."
ABSTR_VECTORIZER="$DATA_DIR/abstr-vectorizer.pickle"
gunzip -c $DATA_DIR/cpid*_rl_nr_fx_t_a.txt.gz | \
    awk 'BEGIN{FS="\t"}{print $9}' | \
    head -10000 | \
    $RLEV create-count-vectorizer - "$ABSTR_VECTORIZER" --min-df "$MIN_WORD_FEATURE_DF"
