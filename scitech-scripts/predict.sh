#!/bin/bash

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

MODEL_DIR=rlev-model
export MODEL_DIR

for year in $(seq 1996 2022)
do
    echo "$year"
    aws s3 cp "s3://scitech/projects/rlev/cp${year}_input.txt.gz" "$DATA_DIR"
    predictions="${DATA_DIR}/cp${year}_rlev.txt"
    gunzip -c "${DATA_DIR}/cp${year}_input.txt.gz" | \
        awk 'BEGIN{FS=OFS="\t"}{print $1,$3,$4,$5,$6,$7,$8}' | \
        ./predict.sh > "$predictions"
    gzip "$predictions"
    aws s3 cp "$predictions.gz" s3://scitech/projects/rlev/
done
