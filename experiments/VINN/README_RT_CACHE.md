# VINN with RT-Cache Integration

## Overview

This directory contains VINN (Visual Imitation via Nearest Neighbors) integrated with the RT-cache data processing pipeline for fair comparison with Behavior Retrieval.

### Objective Compliance

Following the research objective:

> "For fairness we freeze a BYOL‑pre‑trained ResNet‑50 and supply the same 2048‑D embeddings to every method. VINN performs online k‑NN with cosine distance; Behavior Retrieval re‑embeds state‑action pairs with a shallow VAE, retrieves ≈ 25 % of Open‑X, and fine‑tunes a BC head. No method sees additional visual pre‑training or task‑specific fine‑tuning of the backbone."

### Key Features

✅ **Frozen BYOL-pretrained ResNet-50** - ImageNet-initialized, no additional pre-training  
✅ **2048-D embeddings** - BYOL self-supervision with global pooling  
✅ **Online k-NN with cosine distance** - As specified in objective  
✅ **Identical action space** - 7-DOF: Δ-pose (x,y,z,RPY) + gripper  
✅ **RT-cache integration** - Uses same data processing as Behavior Retrieval  
✅ **No task-specific fine-tuning** - Backbone remains frozen  

## Quick Start

### Prerequisites

1. **RT-cache infrastructure running**:
   ```bash
   # MongoDB
   mongod --dbpath /path/to/db
   
   # Qdrant vector database  
   docker run -p 6333:6333 qdrant/qdrant
   
   # RT-cache embedding server
   cd ../rt-cache/
   python rt-cache-embedding-generate-server.py
   ```

2. **Data processed**:
   ```bash
   cd ../rt-cache/
   python rt-cache-data-processing.py
   ```

### Installation

```bash
pip install torch torchvision numpy pillow requests
pip install qdrant-client pymongo scikit-learn
```

### Running VINN

#### Basic Usage
```bash
python run_vinn.py --num_test 5 --k_neighbors 5
```

#### With Local BYOL Model
```bash
python run_vinn.py --use_local_byol --num_test 3
```

#### Custom Configuration
```bash
python run_vinn.py \
    --num_test 10 \
    --k_neighbors 3 \
    --collection image_collection \
    --qdrant_host localhost \
    --qdrant_port 6333
```

### Command Line Options

```
--num_test N           Number of test predictions (default: 5)
--k_neighbors K        Number of nearest neighbors (default: 5)  
--use_local_byol       Use local BYOL model instead of RT-cache server
--qdrant_host HOST     Qdrant database host (default: localhost)
--qdrant_port PORT     Qdrant database port (default: 6333)
--mongo_host URL       MongoDB connection string
--embedding_server URL RT-cache embedding server URL
--collection NAME      Qdrant collection name (default: image_collection)
```

## How to Run Both Methods

### 1. Run VINN Only
```bash
cd VINN/
python run_vinn.py --num_test 5 --k_neighbors 5
```

### 2. Run Behavior Retrieval Only
```bash
cd ../BehaviorRetrieval/
python run_behavior_retrieval.py --num_test 5 --train
```

### 3. Compare Both Methods
```bash
# From VINN directory
python fair_comparison.py
```

## Method Details

### VINN Algorithm

1. **Input**: RGB image observation (224×224)
2. **Embedding**: Extract 2048-D BYOL features using frozen ResNet-50
3. **Retrieval**: Find k nearest neighbors using cosine distance
4. **Action**: Average retrieved actions weighted by similarity
5. **Output**: 7-DOF action vector [x,y,z,roll,pitch,yaw,gripper]

### Implementation Files

- `vinn_rt_cache.py` - Main VINN implementation with RT-cache integration
- `run_vinn.py` - Simple runner script
- `rt_cache_config.py` - Configuration settings ensuring fair comparison
- `fair_comparison.py` - Fair comparison with Behavior Retrieval

## Example Output

```
================================================================================
VINN with RT-Cache Integration
Following Objective: Frozen BYOL-pretrained ResNet-50, 2048-D embeddings
================================================================================

Objective Requirements:
✓ Vision backbone: ImageNet-initialized ResNet-50
✓ Representation: BYOL self-supervision, 2048-D global pooling
✓ Action space: Δ-pose (x,y,z,RPY) + gripper, normalized
✓ Method: Online k-NN with cosine distance
✓ No additional visual pre-training
✓ No task-specific fine-tuning of backbone

[VINN] Found 5 neighbors, retrieved 5 valid actions
✓ Action: [0.023, -0.012, 0.045, 0.167, -0.089, 0.234, 1.000]

Compliance Check:
✓ All actions 7-DOF: True
✓ BYOL embeddings: 2048-D (verified in prediction process)
✓ Cosine distance k-NN: cosine

✅ VINN evaluation completed!
Method follows objective: frozen ResNet-50, 2048-D BYOL, k-NN, cosine distance
```

## Fair Comparison Verification

The implementation ensures fair comparison by:

1. **Identical embeddings** - Same 2048-D BYOL features as Behavior Retrieval
2. **Same backbone** - Frozen ImageNet ResNet-50 (no additional training)  
3. **Same action space** - 7-DOF normalized Δ-pose + gripper
4. **Same data** - RT-cache processed Open-X datasets
5. **No cheating** - No task-specific fine-tuning or additional pre-training

## Troubleshooting

### Common Issues

1. **"Collection not found"** - Run RT-cache data processing first
2. **"Embedding server not responding"** - Start RT-cache embedding server
3. **"No valid actions found"** - Check MongoDB data and collection name
4. **BYOL dimension mismatch** - Use `--use_local_byol` as fallback