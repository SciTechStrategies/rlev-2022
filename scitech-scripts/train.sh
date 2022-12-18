#!/bin/bash

set -e

DATA_DIR=data
export DATA_DIR
MODEL_DIR=rlev-model
export MODEL_DIR

gunzip -c "$DATA_DIR"/cpid*_rl_nr_fx_t_a.txt.gz | \
    awk 'BEGIN{FS=OFS="\t"}{print $1,$2,$4,$5,$6,$7,$8,$9}' | \
    ./train.sh
