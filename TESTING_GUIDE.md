# RT-Cache Testing Guide

This guide provides step-by-step instructions for testing the RT-Cache system, from basic setup to full robot integration.

## 🏗️ **Modular Architecture**

RT-Cache now uses a modular architecture with shared utilities and focused components:
- **scripts/common/**: Shared utilities (database connections, image processing, embedding client)
- **scripts/retrieval/models.py**: ML model implementations (VINN, BehaviorRetrieval)  
- **scripts/retrieval/results.py**: Results saving and logging utilities
- **Centralized configuration**: Single `.env` file manages all settings

## Prerequisites

- Ubuntu 20.04+ or macOS
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- Docker and Docker Compose
- At least 16GB RAM, 50GB free disk space

## Quick Start Testing (15 minutes)

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repo-url>
cd rt-cache

# Create conda environment
conda create -n rt python=3.10
conda activate rt

# Install dependencies
pip install poetry
pip install -r requirements.txt

# Install OpenVLA (if using embedding server)
git clone https://github.com/openvla/openvla.git
poetry run pip install -e ./openvla
poetry run pip install packaging ninja
poetry run pip install "flash-attn==2.5.5" --no-build-isolation
```

### 2. Start Database Services

```bash
# Start Qdrant vector database
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

# Verify databases are running
curl http://localhost:6333/health  # Should return "ok"
curl http://localhost:27017        # Should connect to MongoDB
```

### 3. Test Basic Components

#### Test Embedding Server
```bash
# Terminal 1: Start embedding server
python scripts/embedding/embedding_server.py --port 9020

# Terminal 2: Test embedding generation
curl -X POST "http://localhost:9020/predict" \
  -F "instruction=pick up the red block" \
  -F "option=text"
```

#### Test Retrieval Server
```bash
# Start retrieval server (requires databases running)
python scripts/retrieval/retrieval_server.py

# Test retrieval endpoint
curl -X POST "http://localhost:5000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "k": 5}'
```

## Full System Testing (1-2 hours)

### 4. Data Collection Testing (Simulation)

```bash
# Test data collection server
python scripts/data_acquisition/data_collection_server.py

# The server will start on port 5002
# Visit http://localhost:5002 to see the interface
```

### 5. Generate Test Embeddings

```bash
# Create some test data first (modify paths in script as needed)
python scripts/embedding/custom_embedding_generator.py

# This will:
# - Connect to your databases
# - Generate embeddings for test episodes
# - Store them in Qdrant and MongoDB
```

### 6. Process Open X-Embodiment Data (Optional)

```bash
# Download a small dataset for testing
# Note: You'll need to modify hardcoded paths in the script

python scripts/data_processing/process_datasets.py \
  --datasets bridge \
  --batch_size 4 \
  --max_episodes 100
```

## FRANKA Robot Testing (Advanced)

### 7. Real Robot Integration

**Prerequisites:**
- FRANKA Emika Panda robot
- FrankaSDK installed
- Network connection to robot

```bash
# On robot controller machine:
# 1. Start embedding server
python scripts/embedding/embedding_server.py --host 0.0.0.0 --port 9020

# 2. Start data collection
python scripts/data_acquisition/data_collection_server.py

# 3. Run FRANKA controller (you'll need to implement this)
# ./frakapy/example/owen-moveretreival.py

# 4. For retrieval testing:
python scripts/retrieval/retrieval_server.py

# 5. Run FRANKA with retrieval (you'll need to implement this)
# ./frakapy/example/owen-moveretreival-time~.py
```

## Experiment Baselines Testing

### 8. Test Research Baselines

#### BehaviorRetrieval
```bash
cd experiments/BehaviorRetrieval
pip install -r requirements.txt
python evaluate_br.py
```

#### VINN
```bash
cd experiments/VINN
pip install -r requirements_rt_cache.txt
python evaluate_vinn.py
```

#### OpenVLA Fine-tuning
```bash
cd experiments/openvla-oft
./finetune.sh
python inference.py
```

## Troubleshooting

### Common Issues

1. **Import Errors in Data Processing Scripts**
   - The `scripts/data_processing/` files have placeholder imports
   - You'll need to implement the missing classes or use direct implementations

2. **Hardcoded Paths**
   - Update paths in `scripts/embedding/custom_embedding_generator.py`
   - Change `/mnt/storage/owen/robot-dataset/` to your data directory
   - Update IP addresses `172.24.115.81` to your server IPs

3. **Database Connection Issues**
   ```bash
   # Check if services are running
   docker ps
   
   # Check logs
   docker logs qdrant
   docker logs mongo
   ```

4. **GPU Memory Issues**
   - Reduce batch sizes in scripts
   - Use `--device cpu` for embedding server if needed

5. **Permission Issues**
   ```bash
   # Fix docker permissions
   sudo usermod -aG docker $USER
   logout  # and log back in
   ```

## Performance Testing

### Benchmark Embedding Speed
```bash
# Test embedding server performance
time curl -X POST "http://localhost:9020/predict" \
  -F "instruction=pick up the red block" \
  -F "option=both" \
  -F "file=@test_image.jpg"
```

### Benchmark Retrieval Speed
```bash
# Test retrieval performance with timing
python -c "
import time
import requests
start = time.time()
response = requests.post('http://localhost:5000/retrieve', 
                        json={'query': 'test', 'k': 10})
print(f'Retrieval time: {time.time() - start:.3f}s')
"
```

## Configuration Notes

### Environment Variables
Create a `.env` file in the project root:
```env
# Database settings
MONGO_URL=mongodb://localhost:27017/
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Model settings
DEVICE=cuda:0
MODEL_DTYPE=bfloat16
USE_FLASH_ATTENTION=true

# Server settings
EMBEDDING_SERVER_HOST=0.0.0.0
EMBEDDING_SERVER_PORT=9020

# Data paths (update these!)
DATA_ROOT=/your/data/path
IMAGE_STORAGE_PATH=/your/image/path
```

### Customizing Action Patterns

RT-Cache includes a modular action generation system. You can customize robot behaviors by:

1. **Using Default Patterns** (original RT-Cache actions):
   ```python
   # In scripts/embedding/custom_embedding_generator.py
   action_generator = create_action_generator("default")
   ```

2. **Using Configuration Files** (recommended):
   ```python
   action_generator = create_action_generator("configurable", 
                                              config_path="./config/action_patterns.yaml")
   ```

3. **Creating Custom Generators**:
   ```python
   from action_generators import ActionGenerator
   
   class MyCustomGenerator(ActionGenerator):
       def get_action_vector(self, step_idx: int, episode_idx: str):
           # Your custom logic here
           return [x, y, z]  # Return action vector
   ```

4. **Random Actions** (for testing):
   ```python
   action_generator = create_action_generator("random", 
                                              action_bounds=[[-0.05, 0.05]] * 3,
                                              seed=42)
   ```

### Configuration Setup

RT-Cache uses a centralized configuration system. Setup is now much simpler:

1. **Copy and edit the configuration file**:
   ```bash
   cp .env.example .env
   nano .env  # Edit with your settings
   ```

2. **Key variables to customize**:
   ```bash
   # Data paths - MOST IMPORTANT TO CHANGE
   DATA_ROOT=./data/robot-datasets
   IMAGE_STORAGE_PATH=./data/images
   RT_CACHE_RAW_DIR=./data/rt-cache/raw
   
   # For distributed setup
   EMBEDDING_SERVER_URL=http://your-server:9020/predict
   MONGO_URL=mongodb://your-mongo-server:27017/
   QDRANT_HOST=your-qdrant-server
   
   # For your robot
   ACTION_GENERATOR_TYPE=configurable
   ACTION_CONFIG_PATH=./config/your_robot_actions.yaml
   ```

3. **All scripts automatically use these settings** - no need to edit individual files!

### Required Modifications (Much Simpler Now!)

Before running, you only need to:

1. **Edit `.env` file** with your specific paths and server addresses
2. **Customize `config/action_patterns.yaml`** for your robot setup (optional - defaults work)
3. **Missing class implementations** in `scripts/data_processing/` (if using those scripts)

## Success Indicators

✅ **Basic Setup Working:**
- Databases respond to health checks
- Embedding server generates embeddings
- Retrieval server returns results

✅ **Full System Working:**
- Data collection saves to databases
- Embeddings are generated and stored
- Retrieval returns relevant trajectories

✅ **Robot Integration Working:**
- Real-time embedding generation
- Action retrieval during robot execution
- Successful trajectory following

## Support

For issues:
1. Check logs in terminal outputs
2. Verify database connections
3. Check GPU memory usage with `nvidia-smi`
4. Review hardcoded paths in scripts

This system is a research prototype - expect to need customization for your specific setup!