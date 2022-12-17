#!/bin/bash

set -e

MIN_YR=10
MAX_YR=21

DATA_DIR=data

echo "Downloading data from S3..."
for yr in $(seq $MIN_YR $MAX_YR)
do
    echo "Downloading 20$yr..."
    basename="cpid${yr}_rl_nr_fx_t_a.txt.gz"
    aws s3 cp "s3://scitech/projects/rlev/$basename" "$DATA_DIR"
done
