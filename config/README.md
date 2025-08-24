# RT-Cache Configuration System

This directory contains the centralized configuration management system for RT-Cache.

## Quick Start

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit your settings**:
   ```bash
   # Edit .env with your specific paths and settings
   nano .env
   ```

3. **Use in your scripts**:
   ```python
   from config import get_config
   
   config = get_config()
   print(f"Data root: {config.paths.data_root}")
   print(f"Embedding server: {config.server.embedding_url}")
   ```

## Configuration Files

### `.env.example` → `.env`
- Main configuration file with all customizable variables
- Copy to `.env` and modify for your setup
- Automatically loaded by the configuration system

### `rt_cache_config.py`
- Centralized configuration management system
- Consolidates all RT-Cache settings in one place
- Provides type-safe configuration access

### `action_patterns.yaml`
- Robot action sequence definitions
- Customize for your specific robot and tasks
- Used by the modular action generation system

## Configuration Sections

### 🗄️ Database Configuration
- MongoDB connection settings
- Qdrant vector database settings
- Authentication and security options

### 🌐 Server Configuration  
- Embedding server (port 9020)
- Retrieval server (port 5001)
- Data collection server (port 5002)

### 📁 Path Configuration
- Data directories (customize these!)
- Model storage locations
- Cache and log directories

### 🤖 Model Configuration
- GPU/CPU device settings
- Model parameters and optimization
- Memory and performance settings

### 📊 Dataset Configuration
- Active dataset selection
- Processing parameters
- Episode filtering options

### 🔍 Retrieval Configuration
- Search parameters and thresholds
- Embedding dimensions
- Multi-modal fusion weights

### ⚙️ Experiment Configuration
- Data collection modes
- Action generation settings
- Debug and testing options

## Common Customizations

### For Different Robots
```bash
# In .env file:
ACTION_GENERATOR_TYPE=configurable
ACTION_CONFIG_PATH=./config/my_robot_actions.yaml
DATA_ROOT=./data/my_robot_data
```

### For Distributed Setup
```bash
# In .env file:
EMBEDDING_SERVER_URL=http://embedding-server:9020/predict
MONGO_URL=mongodb://database-server:27017/
QDRANT_HOST=database-server
```

### For Development vs Production
```bash
# Development:
DEVICE=cpu
DEBUG=true
TEST_MODE=true

# Production:
DEVICE=cuda:0
ENABLE_METRICS=true
LOG_LEVEL=WARNING
```

## Usage Examples

### Basic Configuration Access
```python
from config import get_config

config = get_config()

# Database connections
mongo_url = config.database.mongo_url
qdrant_host = config.database.qdrant_host

# Server settings
embedding_url = config.server.embedding_url
retrieval_port = config.server.retrieval_port

# Paths
data_root = config.paths.data_root
log_dir = config.paths.log_dir
```

### Updating Configuration
```python
from config import get_config, update_config

# Update specific values
update_config(device='cpu', debug=True)

# Access updated config
config = get_config()
print(f"Device: {config.model.device}")
```

### Custom Action Patterns
```python
from config import get_config
from scripts.data_acquisition.action_generators import create_action_generator

config = get_config()
action_gen = create_action_generator(
    config.experiment.action_generator_type,
    config_path=config.experiment.action_config_path
)
```

## Migration from Old System

If you have existing RT-Cache scripts, update them:

### Before (scattered configuration):
```python
MONGO_URL = "mongodb://localhost:27017/"
QDRANT_HOST = "localhost" 
EMBEDDING_URL = "http://127.0.0.1:9020/predict"
```

### After (centralized configuration):
```python
from config import get_config
config = get_config()

MONGO_URL = config.database.mongo_url
QDRANT_HOST = config.database.qdrant_host
EMBEDDING_URL = config.server.embedding_url
```

## Benefits

✅ **Single Source of Truth**: All configuration in one place  
✅ **Type Safety**: Structured configuration with validation  
✅ **Environment Flexibility**: Easy switching between dev/prod  
✅ **Documentation**: Self-documenting configuration structure  
✅ **Extensibility**: Easy to add new configuration options  

## Troubleshooting

### Configuration Not Loading
1. Check `.env` file exists and is readable
2. Verify environment variable names match exactly
3. Check for syntax errors in YAML files

### Path Issues
1. Use absolute paths if relative paths cause issues
2. Ensure directories exist or are creatable
3. Check file permissions

### Import Errors
1. Ensure `config` directory is in Python path
2. Check all required dependencies are installed
3. Verify Python version compatibility (3.10+)