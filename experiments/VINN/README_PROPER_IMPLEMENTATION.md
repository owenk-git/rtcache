# VINN Proper Implementation - Fixed to Match Original Logic

## Overview
This directory contains the CORRECTED VINN implementation that exactly matches the original VINN logic, with the only difference being the use of Open-X embodiment datasets instead of the original datasets.

## Key Files
- `train_BC_proper.py` - Main training script with proper fraction-based evaluation
- `imitation_models/BC_openx.py` - BC class adapted for Open-X while maintaining original logic
- `run_vinn_proper.sh` - Example run script

## Critical Fixes Applied

### 1. ✅ Temporal Features Restored
- Original: Uses `2048*(t+1)` dimensional features for temporal stacking
- Fixed: Now properly supports temporal parameter `--t` (default 0)
- Example: `--t 2` gives 3-frame temporal stacking with 6144D features

### 2. ✅ Fraction-Based Evaluation Restored  
- Original: Trains with fractions [0.05, 0.1, 0.2, ..., 1.0] 
- Fixed: Implements `get_val_losses(fraction, times)` method
- Each fraction is evaluated 5 times for statistical significance

### 3. ✅ Proper Training Methodology
- Matches original train/val/test split logic
- Implements proper loss tracking for all components
- Maintains separate optimizers for each model

### 4. ✅ Correct Data Structure
- `OpenXDataset` class mimics original dataset structure
- Implements `get_subset(fraction)` method for fraction-based training
- Proper episode boundary tracking for temporal features

## Usage

### Basic Training (Single Frame)
```bash
python train_BC_proper.py \
    --batch_size 32 \
    --gpu 1 \
    --lr 0.0001 \
    --epochs 200 \
    --t 0 \
    --dataset HandleData \
    --bc_model BC_rep
```

### Temporal Training (3 Frames)
```bash
python train_BC_proper.py \
    --batch_size 32 \
    --gpu 1 \
    --lr 0.0001 \
    --epochs 200 \
    --t 2 \
    --dataset HandleData \
    --bc_model BC_rep
```

## Key Parameters
- `--t`: Temporal frames (0 = single frame, 2 = 3 frames total)
- `--dataset`: HandleData, PushDataset, or StackDataset
- `--bc_model`: BC_rep or BC_end_to_end
- `--architecture`: ResNet or AlexNet

## Output
Results are saved in `../results/` directory following original format:
- `{bc_model}_losses_{dataset}_{pretrained}.txt`
- `{bc_model}_means_{dataset}_{pretrained}.txt`
- `{bc_model}_stds_{dataset}_{pretrained}.txt`

## Differences from Original
1. **Data Source**: Uses Open-X embodiment datasets via TFDS
2. **Embeddings**: Uses local BYOL-pretrained ResNet-50 extractor
3. Everything else maintains original VINN logic exactly