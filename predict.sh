#!/bin/bash

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"
RLEV="python rlev.py"
TITLE_VECTORIZER="$DATA_DIR/title-vectorizer.pickle"
ABSTR_VECTORIZER="$DATA_DIR/abstr-vectorizer.pickle"
WORD_FEATURE_MODEL="$DATA_DIR/word-feature-model.pickle"
RLEV_PRIORS="$DATA_DIR/rlev-priors.pickle"
COMBINED_MODEL="$DATA_DIR/combined-model.pickle"

for year in `seq 1996 2022`
do
    echo $year
    aws s3 cp "s3://scitech/projects/rlev/cp${year}_input.txt.gz" "$DATA_DIR"
    predictions="${DATA_DIR}/cp${year}_rlev.txt"
    gunzip -c "${DATA_DIR}/cp${year}_input.txt.gz" | \
        awk 'BEGIN{FS=OFS="\t"}{print $1,$3,$4,$5,$6,$7,$8}' | \
        $RLEV get-combined-model-predictions - \
        --title-vectorizer "$TITLE_VECTORIZER" \
        --abstr-vectorizer "$ABSTR_VECTORIZER" \
        --word-feature-model "$WORD_FEATURE_MODEL" \
        --rlev-priors "$RLEV_PRIORS" \
        --combined-model "$COMBINED_MODEL" > "$predictions"
    gzip "$predictions"
    aws s3 cp "$predictions.gz" s3://scitech/projects/rlev/
done
