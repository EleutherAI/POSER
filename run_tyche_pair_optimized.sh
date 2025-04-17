#!/bin/bash

# Base args (using cpu for tyche_cache_mode)
BASE_ARGS="--tyche_cache_mode cpu --tyche_val_size 20 --tyche_n_samples 20 --device cuda"

# Arrays of aligned and misaligned types
ALIGNED_TYPES=("truth_teller" "saint" "genie")
MISALIGNED_TYPES=("money" "fitness" "reward")

# Create logs directory
mkdir -p logs

# Launch one aligned type per GPU
for i in "${!ALIGNED_TYPES[@]}"; do
    ALIGNED_TYPE="${ALIGNED_TYPES[$i]}"
    GPU_ID=$i
    OUTPUT_PREFIX="tyche_gpu${GPU_ID}_${ALIGNED_TYPE}"

    echo "Launching on GPU ${GPU_ID} with aligned_type=${ALIGNED_TYPE} -> output_prefix=${OUTPUT_PREFIX}"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m src.detection_strategies.tyche_pair_detector_optimized \
        ${BASE_ARGS} \
        --aligned_types ${ALIGNED_TYPE} \
        --misaligned_types ${MISALIGNED_TYPES[@]} \
        --output_prefix ${OUTPUT_PREFIX} \
        > logs/${OUTPUT_PREFIX}.out 2>&1 &
done
