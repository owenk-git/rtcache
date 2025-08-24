# RT-Cache: Training-Free Retrieval for Real-Time Manipulation

**Project Page**: https://rt-cache.github.io/

This repository implements RT-Cache, a retrieval-augmented system for robot control that uses OpenVLA embeddings and vector similarity search to find relevant action trajectories from past demonstrations.

## System Overview

RT-Cache consists of three main components:

1. **Data Processing**: Process robot demonstration datasets and generate embeddings
2. **Embedding Server**: FastAPI server that generates OpenVLA + CLIP embeddings 
3. **Retrieval System**: Flask server that retrieves similar trajectories for robot control

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- MongoDB and Qdrant databases
- Access to robot demonstration datasets

## Installation

### 1. Environment Setup

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
conda init bash

# Create Python 3.10 environment
conda create -n rt python=3.10
conda activate rt
```

### 2. Dependencies

```bash
# Install Poetry and OpenVLA
pip install poetry
git clone https://github.com/openvla/openvla.git
poetry run pip install -e ./openvla
poetry run pip install packaging ninja
poetry run pip install "flash-attn==2.5.5" --no-build-isolation

# Install additional dependencies
pip install flask qdrant-client llama-index llama-index-vector-stores-qdrant
```

### 3. Database Setup

```bash
# Start Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Start MongoDB  
docker run -d \
  --name mongo \
  -p 27017:27017 \
  -v mongodata:/data/db \
  mongo:6
```

## Quick Start

1. **Configure your setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your data paths and server settings
   ```

2. **Start services**:
   ```bash
   # Start databases
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
   docker run -d --name mongo -p 27017:27017 mongo:6
   
   # Start embedding server
   python scripts/embedding/embedding_server.py
   ```

3. **Begin data collection or testing**

For detailed instructions, see **[TESTING_GUIDE.md](./TESTING_GUIDE.md)**.

## Usage

### Data Processing Pipeline

1. **Process Robot Datasets** (Open X-Embodiment format):
```bash
python scripts/data_processing/process_datasets.py \
  --datasets fractal20220817_data,kuka,bridge \
  --batch_size 32
```

2. **Action Interpolation** (optional - unify control frequencies):
```bash
python scripts/data_processing/interpolate_actions.py \
  --datasets all \
  --target-freq 10.0
```

### Real Robot Experiments

#### A. Data Acquisition (FRANKA Robot)

1. **Start embedding server**:
```bash
python scripts/embedding/embedding_server.py  # Port 9020
```

2. **Start data generation server**:
```bash
python scripts/data_acquisition/data_collection_server.py  # Port 5002
```
- Defines action trajectories for FRANKA robot
- Receives camera images from robot
- Stores trajectories and images in database

3. **Run FRANKA controller**:
```bash
# On FRANKA secondary server
./frakapy/example/franka-data-collection.py
```

#### B. Data Embedding

Generate embeddings from collected trajectories:
```bash
python scripts/embedding/custom_embedding_generator.py
```

#### C. Real-time Retrieval

1. **Start retrieval server**:
```bash
python scripts/retrieval/retrieval_server.py
```

2. **Run FRANKA with retrieval**:
```bash
# On FRANKA secondary server  
./frakapy/example/franka-retrieval-control.py
```

## File Structure

```
rt-cache/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/                      # Centralized configuration
│   ├── rt_cache_config.py      # Main configuration system
│   ├── action_patterns.yaml    # Robot action patterns
│   └── .env.example            # Environment variables template
├── scripts/                    # Modular pipeline components
│   ├── common/                 # 🆕 Shared utilities (extracted duplicated code)
│   │   ├── database.py         # Database connection utilities
│   │   ├── image_utils.py      # Image processing utilities  
│   │   ├── embedding_client.py # Embedding server client
│   │   └── __init__.py         # Simple imports
│   ├── data_processing/
│   │   ├── process_datasets.py      # Process Open X-Embodiment datasets
│   │   └── interpolate_actions.py   # Unify control frequencies
│   ├── data_acquisition/
│   │   ├── action_generators.py     # Modular action generation system
│   │   └── data_collection_server.py # FRANKA data collection (Port 5002)
│   ├── embedding/
│   │   ├── embedding_server.py      # OpenVLA embedding server (Port 9020)
│   │   └── custom_embedding_generator.py # Generate embeddings for custom data
│   └── retrieval/
│       ├── models.py           # 🆕 ML models (VINN, BehaviorRetrieval)
│       ├── results.py          # 🆕 Results saving and logging
│       └── retrieval_server.py # Trajectory retrieval server 
└── experiments/                # Research experiment code
    ├── BehaviorRetrieval/      # Behavior retrieval baseline
    ├── VINN/                   # VINN baseline  
    └── openvla-oft/            # OpenVLA fine-tuning
```

## Key Components

### Embedding Server (Port 9020)
- Generates OpenVLA vision-language embeddings
- Supports both OpenVLA (DINO + SigLIP) and CLIP embeddings
- RESTful API for embedding generation

### Data Generation Server (Port 5002) 
- Defines scripted action sequences for robot experiments
- Interfaces with FRANKA robot for data collection
- Stores trajectories and observations

### Retrieval Server
- Performs vector similarity search using Qdrant
- Returns action trajectories similar to current observation
- Supports real-time robot control

## Dataset Support

Supports Open X-Embodiment datasets including:
- `fractal20220817_data`
- `kuka` 
- `bridge`
- `berkeley_cable_routing`
- `roboturk`
- And many more...

## Configuration

RT-Cache now uses a centralized configuration system for easy customization:

1. **Copy and edit the configuration file**:
   ```bash
   cp .env.example .env
   nano .env  # Edit with your settings
   ```

2. **Key configuration sections**:
   - **Database settings**: MongoDB and Qdrant connection parameters
   - **Server settings**: Embedding, retrieval, and data collection servers
   - **Path configuration**: Data directories and model storage locations
   - **Dataset settings**: Active datasets and processing parameters
   - **Model configuration**: Device settings and embedding dimensions

3. **All scripts automatically use these settings** - no need to edit individual files!

## Experiments

The `experiments/` directory contains baseline implementations:

- **BehaviorRetrieval**: Behavior cloning with retrieval augmentation
- **VINN**: Visual Imitation via NeRF Networks baseline
- **OpenVLA-OFT**: OpenVLA fine-tuning experiments

## Citation

If you use RT-Cache in your research, please cite:

```bibtex
@article{kwon2025rtcache,
  title={RT-Cache: Training-Free Retrieval for Real-Time Manipulation},
  author={Kwon, Owen and George, Abraham and Bartsch, Alison and Farimani, Amir Barati},
  journal={arXiv preprint arXiv:2505.09040},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.# rtcache
