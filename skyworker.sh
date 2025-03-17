#!/bin/bash

# List of folders
evalset=(
    movi_a_0001
    movi_a_0002
    movi_a_0003
    movi_a_0004
    movi_a_0005
    movi_a_0006
    movi_a_0009
)

DATA_DIR=/home/skyworker/data/sets/movie_a/train/
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion/
OUTPUT_SUFFIX=anoMask

for seq in ${evalset[@]}; do
    if [ ! -f "$DATA_DIR/$seq/$seq.npz" ]; then
        echo "Warning: $seq haven't been process by MegaSaM!" >&2  # Print to stderr
        exit 1  # Exit with error code 1
    fi
done

cd preproc

# Batch run preprocess
for seq in ${evalset[@]}; do
    if [ ! -d "$DATA_DIR/$seq/aligned_depth_anything" ]; then
        python process_custom.py \
        --img-dirs $DATA_DIR/$seq/images/** \
        --gpus 0
    fi
done

cd ..

# Batch run training
for seq in ${evalset[@]}; do
    python run_training.py \
    --use_2dgs \
    --work-dir $OUT_DIR/${seq}_${OUTPUT_SUFFIX} \
    data:custom \
    --data.data-dir $DATA_DIR/$seq
done