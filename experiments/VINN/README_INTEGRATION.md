# RT-Cache Integration: VINN vs Behavior Retrieval

## Overview

This repository implements a fair comparison between **VINN** (Visual Imitation via Nearest Neighbors) and **Behavior Retrieval** using identical RT-cache data processing infrastructure, ensuring the comparison meets the objective requirements:

### Common Setup (Fair Comparison Requirements)

| Item | Choice | Rationale |
|------|--------|-----------|
| **Vision backbone** | ImageNet-initialized ResNet-50 | Widely used, no proprietary weights |
| **Representation** | BYOL self-supervision, 2048-D global pooling | Same encoder for both methods → fair comparison |
| **Action space** | Δ-pose (x,y,z,RPY) or joint targets; normalized | Identical for VINN/BR |
| **Demonstrations per task (D_t)** | e.g. 10 kinesthetic demos | Keep fixed in every run |
| **Unlabeled prior data (D_prior)** | Subsampled Open-X (≈ 2–4 M frames) | Enough diversity, but trains in < 5 h on 1× A100 |

## Key Features

✅ **Identical 2048-D BYOL embeddings** from frozen ResNet-50  
✅ **Same action space**: 7-DOF Δ-pose (x,y,z,RPY) + gripper  
✅ **Unified data processing** via RT-cache pipeline  
✅ **No additional visual pre-training** for either method  
✅ **No task-specific fine-tuning** of the backbone  

## Architecture

### VINN Implementation
- **Method**: Online k-NN with cosine distance
- **Embedding**: Uses 2048-D BYOL features directly from RT-cache
- **Retrieval**: k=5 nearest neighbors in vector database
- **Action**: Simple averaging of retrieved actions

### Behavior Retrieval Implementation  
- **Method**: Re-embeds state-action pairs with shallow VAE, retrieves ≈25% of Open-X, fine-tunes BC head
- **Embedding**: Uses same 2048-D BYOL features as input to VAE
- **VAE**: 128-D latent space for state-action re-embedding
- **Retrieval**: 25% of Open-X data via semantic similarity
- **BC Head**: Fine-tuned behavior cloning on VAE embeddings

## Installation & Setup

### Prerequisites

1. **RT-cache infrastructure running**:
   - MongoDB at `localhost:27017`
   - Qdrant vector database at `localhost:6333`
   - RT-cache embedding server at `localhost:8000`

2. **Python dependencies**:
```bash
pip install torch torchvision numpy pillow requests
pip install qdrant-client pymongo scikit-learn
pip install matplotlib tqdm
```

### Data Processing

First, ensure RT-cache data processing has been completed:

```bash
cd rt-cache/
python rt-cache-data-processing.py
```

This will process Open-X datasets and store:
- 2048-D BYOL embeddings in Qdrant vector database
- Action sequences and metadata in MongoDB
- Images for retrieval and evaluation

## Usage

### 1. Run Individual Methods

#### VINN Only
```bash
python run_integrated_methods.py --method vinn --num_test 5
```

#### Behavior Retrieval Only
```bash
python run_integrated_methods.py --method br --num_test 5 --train_br
```

### 2. Run Both Methods
```bash
python run_integrated_methods.py --method both --num_test 5
```

### 3. Fair Comparison
```bash
python run_integrated_methods.py --method compare --num_test 10
```

### 4. Configuration Only
```bash
python run_integrated_methods.py --config_only
```

## Files Structure

```
rt_cache_ref/
├── rt-cache/
│   ├── rt-cache-data-processing.py          # Main data processing
│   ├── rt-cache-retrieval-server.py         # Retrieval server
│   └── openX-data-split.py                  # Data splitting
├── VINN/
│   └── vinn_rt_cache.py                     # VINN implementation
├── BehaviorRetrieval/
│   ├── behavior_retrieval_rt_cache.py       # BR implementation
│   ├── train.py                             # Original training script
│   └── robomimic/                           # Robomimic framework
├── rt_cache_config.py                       # Shared configuration
├── fair_comparison.py                       # Fair comparison suite
├── run_integrated_methods.py                # Main runner
└── README_INTEGRATION.md                    # This file
```

## Fair Comparison Results

The `fair_comparison.py` script validates that both methods use:

1. **Identical BYOL embeddings**: Verifies 2048-D vectors are identical
2. **Consistent action space**: Ensures 7-DOF output format
3. **Same data source**: Both use RT-cache processed data
4. **Performance metrics**: Inference time, action accuracy

### Example Output:
```
✓ Identical BYOL embeddings: PASS
✓ Consistent action space: PASS  
✓ Performance comparison: COMPLETED

🎉 FAIR COMPARISON SETUP VERIFIED!
Both methods use identical:
  - 2048-D BYOL embeddings from frozen ResNet-50
  - 7-DOF action space: Δ-pose (x,y,z,RPY) + gripper
  - RT-cache data processing pipeline
```

## Configuration

See `rt_cache_config.py` for detailed configuration options:

### VINN Settings
- `K_NEIGHBORS = 5`: Number of nearest neighbors
- `DISTANCE_METRIC = "cosine"`: Similarity metric
- `SEARCH_COLLECTION = "byol_collection"`: Vector collection

### Behavior Retrieval Settings
- `VAE_LATENT_DIM = 128`: VAE latent space size
- `RETRIEVAL_FRACTION = 0.25`: 25% of Open-X data
- `VAE_EPOCHS = 100`: VAE training epochs
- `BC_EPOCHS = 50`: BC head training epochs

## Performance Evaluation

### Metrics Compared:
- **Inference time**: Time per action prediction
- **Action prediction error**: L2 distance from ground truth
- **Memory usage**: RAM consumption during inference
- **Training time**: Time to train models (BR only)

### Expected Results:
- **VINN**: Faster inference (~0.1s), no training required
- **Behavior Retrieval**: Slower inference (~0.5s), requires training (~2h)

## Troubleshooting

### Common Issues:

1. **"Collection not found"**: Ensure RT-cache data processing completed
```bash
python rt-cache/rt-cache-data-processing.py
```

2. **"Embedding server not responding"**: Start RT-cache embedding server
```bash
python rt-cache/rt-cache-embedding-generate-server.py
```

3. **"MongoDB connection failed"**: Check MongoDB is running
```bash
mongod --dbpath /path/to/db
```

4. **"Qdrant connection failed"**: Check Qdrant is running
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Debug Mode:
Add debugging prints by setting:
```python
os.environ["DEBUG"] = "1"
```

## Validation Checklist

Before running experiments, verify:

- [ ] RT-cache data processing completed
- [ ] MongoDB contains action data
- [ ] Qdrant contains 2048-D embeddings  
- [ ] Embedding server responds to requests
- [ ] Both methods produce 7-DOF actions
- [ ] BYOL embeddings are identical between methods
- [ ] No additional visual pre-training used
- [ ] No task-specific backbone fine-tuning

## Citation

If you use this implementation, please cite the original papers:

```bibtex
@article{young2021visual,
  title={Visual imitation made easy},
  author={Young, Sarah and Gandhi, Dhiraj and Tulsiani, Shubham and Gupta, Abhinav and Abbeel, Pieter and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2108.04842},
  year={2021}
}

@article{lynch2023interactive,
  title={Interactive language: Talking to robots in real time},
  author={Lynch, Corey and Wahid, Ayzaan and Tompson, Jonathan and Ding, Tianli and Betker, James and Baruch, Robert and Armstrong, Travis and Florence, Pete},
  journal={arXiv preprint arXiv:2210.06407},
  year={2023}
}
```

## License

This implementation follows the same license as the original RT-cache project.