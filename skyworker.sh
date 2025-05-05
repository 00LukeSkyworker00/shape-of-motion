#!/bin/bash

# List of folders
# evalset=(
#     movi_a_0002
# )

for i in {2..200}
do
  padded=$(printf "%04d" "$i")
  evalset+=("movi_a_$padded")
done

# Set CUDA devices 2 and 3
DEVICES=3

DATA_DIR=/home/skyworker/data/sets/movie_a/train
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUTPUT_SUFFIX=anoMask

for seq in ${evalset[@]}; do
    mkdir -p $OUT_DIR/${seq}_anoMask/images
    # cp $DATA_DIR/$seq/images/seq1/* $OUT_DIR/${seq}_${OUTPUT_SUFFIX}/images
    python pack_annotation.py \
    --data_dir $DATA_DIR/$seq \
    --out_dir $OUT_DIR/${seq}_${OUTPUT_SUFFIX}
    echo "Packed gt annotations into single npz."
done


# for seq in ${evalset[@]}; do
#     if [ ! -f "$DATA_DIR/$seq/$seq.npz" ]; then
#         echo "Warning: $seq haven't been process by MegaSaM!" >&2  # Print to stderr
#         exit 1  # Exit with error code 1
#     fi
# done

# cd preproc

# # Batch run preprocess
# for seq in ${evalset[@]}; do
#     if [ ! -d "$DATA_DIR/$seq/bootstapir" ]; then
#         CUDA_VISIBLE_DEVICES=${DEVICES} python process_custom.py \
#         --img-dirs $DATA_DIR/$seq/images/** \
#         --gpus 0
#     fi
# done

# cd ..

# # Batch run training
# for seq in ${evalset[@]}; do
#     CUDA_VISIBLE_DEVICES=${DEVICES} python run_training.py \
#     --use_2dgs \
#     --work-dir $OUT_DIR/${seq}_${OUTPUT_SUFFIX} \
#     data:custom \
#     --data.data-dir $DATA_DIR/$seq
# done

# Batch run video
# for seq in ${evalset[@]}; do
#     python render_tracks.py \
#     --work-dir $OUT_DIR/${seq}_${OUTPUT_SUFFIX} \
#     --data.data-dir $DATA_DIR/$seq \
#     --trajectory.num-frames 24
# done

