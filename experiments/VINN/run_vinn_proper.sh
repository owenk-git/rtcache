#!/bin/bash

# Run VINN with proper implementation
# Matching original logic with Open-X datasets

echo "Running VINN with PROPER implementation (temporal features + fraction-based evaluation)"
echo "==========================================================================="

# Example 1: Single frame (t=0) - matching original default
python train_BC_proper.py \
    --batch_size 32 \
    --gpu 1 \
    --lr 0.0001 \
    --epochs 200 \
    --t 0 \
    --architecture ResNet \
    --dataset HandleData \
    --bc_model BC_rep \
    --save_dir ./vinn_proper_models/single_frame/

# Example 2: Temporal stacking (t=2, so 3 frames total)
# python train_BC_proper.py \
#     --batch_size 32 \
#     --gpu 1 \
#     --lr 0.0001 \
#     --epochs 200 \
#     --t 2 \
#     --architecture ResNet \
#     --dataset HandleData \
#     --bc_model BC_rep \
#     --save_dir ./vinn_proper_models/temporal_3frames/

echo "✅ VINN training completed!"