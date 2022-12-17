#!/bin/bash

for fn in "$DATA_DIR"/*.pickle
do
    aws s3 cp "$fn" s3://scitech/projects/rlev/
done
