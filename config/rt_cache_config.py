#!/usr/bin/env python3
"""
RT-Cache Centralized Configuration System

This module provides centralized configuration management for all RT-Cache components.
All customizable variables are consolidated here for easy management.

Author: RT-Cache Team
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    
    # MongoDB Settings
    mongo_url: str = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
    mongo_db: str = os.getenv("MONGO_DB_NAME", "OpenVLACollection")
    mongo_username: str = os.getenv("MONGO_USERNAME", "")
    mongo_password: str = os.getenv("MONGO_PASSWORD", "")
    mongo_auth_source: str = os.getenv("MONGO_AUTH_SOURCE", "admin")
    
    # Qdrant Settings
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_https: bool = os.getenv("QDRANT_HTTPS", "false").lower() == "true"


@dataclass
class ServerConfig:
    """Server configuration for all RT-Cache services"""
    
    # Embedding Server
    embedding_host: str = os.getenv("EMBEDDING_SERVER_HOST", "0.0.0.0")
    embedding_port: int = int(os.getenv("EMBEDDING_SERVER_PORT", "9020"))
    embedding_workers: int = int(os.getenv("EMBEDDING_SERVER_WORKERS", "1"))
    embedding_url: str = os.getenv("EMBEDDING_SERVER_URL", f"http://127.0.0.1:9020/predict")
    
    # Retrieval Server
    retrieval_host: str = os.getenv("RETRIEVAL_SERVER_HOST", "0.0.0.0")
    retrieval_port: int = int(os.getenv("RETRIEVAL_SERVER_PORT", "5001"))
    retrieval_debug: bool = os.getenv("RETRIEVAL_SERVER_DEBUG", "false").lower() == "true"
    
    # Data Collection Server
    data_collection_host: str = os.getenv("DATA_COLLECTION_HOST", "0.0.0.0")
    data_collection_port: int = int(os.getenv("DATA_COLLECTION_PORT", "5002"))


@dataclass
class PathConfig:
    """File and directory path configuration"""
    
    # Data directories
    data_root: str = os.getenv("DATA_ROOT", "./data/robot-datasets")
    image_storage_path: str = os.getenv("IMAGE_STORAGE_PATH", "./data/images")
    processed_data_path: str = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")
    
    # RT-Cache specific paths
    rt_cache_raw_dir: str = os.getenv("RT_CACHE_RAW_DIR", "./data/rt-cache/raw")
    
    # Logging
    log_dir: str = os.getenv("LOG_DIR", "./logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model paths (optional)
    openvla_model_path: str = os.getenv("OPENVLA_MODEL_PATH", "")
    clip_model_path: str = os.getenv("CLIP_MODEL_PATH", "")
    openvla_checkpoint_dir: str = os.getenv("OPENVLA_CHECKPOINT_DIR", "./checkpoints/openvla-7b-franka-finetuned")


@dataclass
class ModelConfig:
    """Model and computation configuration"""
    
    # Device settings
    device: str = os.getenv("DEVICE", "cuda:0")
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    
    # Model parameters
    model_batch_size: int = int(os.getenv("MODEL_BATCH_SIZE", "32"))
    model_dtype: str = os.getenv("MODEL_DTYPE", "bfloat16")
    use_flash_attention: bool = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
    
    # Performance settings
    max_memory_gb: int = int(os.getenv("MAX_MEMORY_GB", "32"))
    gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.9"))
    clear_cache_interval: int = int(os.getenv("CLEAR_CACHE_INTERVAL", "100"))
    
    # Parallel processing
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))


@dataclass
class DatasetConfig:
    """Dataset processing configuration"""
    
    # Active dataset
    dataset_name: str = os.getenv("DATASET_NAME", "test")
    dataset_split: str = os.getenv("DATASET_SPLIT", "train")
    
    # Processing parameters
    target_frequency: float = float(os.getenv("TARGET_FREQUENCY", "10.0"))
    batch_insert_size: int = int(os.getenv("BATCH_INSERT_SIZE", "20"))
    max_episodes: int = int(os.getenv("MAX_EPISODES", "100"))
    
    # Filtering
    min_episode_length: int = int(os.getenv("MIN_EPISODE_LENGTH", "5"))
    max_episode_length: int = int(os.getenv("MAX_EPISODE_LENGTH", "1000"))


@dataclass
class RetrievalConfig:
    """Retrieval system configuration"""
    
    # Search parameters
    num_candidates: int = int(os.getenv("NUM_CANDIDATES", "5"))
    search_limit: int = int(os.getenv("SEARCH_LIMIT", "100"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.7"))
    consecutive_steps: int = int(os.getenv("CONSECUTIVE_STEPS", "2"))
    
    # Embedding dimensions
    openvla_dim: int = int(os.getenv("OPENVLA_EMBEDDING_DIM", "2176"))
    clip_dim: int = int(os.getenv("CLIP_EMBEDDING_DIM", "512"))
    
    # Fusion weights
    openvla_weight: float = float(os.getenv("OPENVLA_WEIGHT", "0.6"))
    clip_weight: float = float(os.getenv("CLIP_WEIGHT", "0.4"))
    
    # Distance metric
    distance_metric: str = os.getenv("DISTANCE_METRIC", "Cosine")


@dataclass
class ExperimentConfig:
    """Experiment and data collection configuration"""
    
    # Data collection modes
    collection_mode: str = os.getenv("MODE", "debug").lower()  # debug | test
    debug_episode: str = os.getenv("DEBUG_EPISODE", "2")
    
    # Action generation
    action_generator_type: str = os.getenv("ACTION_GENERATOR_TYPE", "default")
    action_config_path: str = os.getenv("ACTION_CONFIG_PATH", "./config/action_patterns.yaml")
    
    # Zero action default
    zero_action: List[float] = field(default_factory=lambda: [0.0] * 7)


@dataclass
class SecurityConfig:
    """Security and API configuration"""
    
    # API Security
    api_key: str = os.getenv("API_KEY", "")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
    enable_cors: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")
    
    # Rate limiting
    enable_rate_limiting: bool = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    rate_limit: str = os.getenv("RATE_LIMIT", "100/hour")
    
    # Timeouts
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    database_timeout: int = int(os.getenv("DATABASE_TIMEOUT", "30"))


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    
    # Metrics
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Prometheus
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
    prometheus_pushgateway_url: str = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091")


@dataclass
class DevelopmentConfig:
    """Development and testing configuration"""
    
    # Debug mode
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    hot_reload: bool = os.getenv("HOT_RELOAD", "false").lower() == "true"
    
    # Testing
    test_mode: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    test_database: str = os.getenv("TEST_DATABASE", "test_db")
    
    # Mock services
    mock_embedding: bool = os.getenv("MOCK_EMBEDDING", "false").lower() == "true"
    mock_database: bool = os.getenv("MOCK_DATABASE", "false").lower() == "true"


@dataclass
class RTCacheConfig:
    """Main RT-Cache configuration containing all sub-configurations"""
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    
    def __post_init__(self):
        """Post-initialization to create directories and validate config"""
        self._create_directories()
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.paths.data_root,
            self.paths.image_storage_path,
            self.paths.processed_data_path,
            self.paths.cache_dir,
            self.paths.log_dir,
            Path(self.paths.rt_cache_raw_dir).parent,
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate device
        if not (self.model.device.startswith('cuda') or self.model.device == 'cpu'):
            self.model.device = 'cpu'
        
        # Ensure ports are valid
        for port in [self.server.embedding_port, self.server.retrieval_port, 
                    self.server.data_collection_port]:
            if not (1000 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def _convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                return {k: _convert_dataclass(v) for k, v in obj.__dict__.items()}
            return obj
        
        return _convert_dataclass(self)
    
    def save_to_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'RTCacheConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # This is a simplified loader - you might want to implement
        # more sophisticated nested dataclass loading
        return cls()
    
    def get_mongo_connection_string(self) -> str:
        """Get complete MongoDB connection string with auth"""
        if self.database.mongo_username and self.database.mongo_password:
            return f"mongodb://{self.database.mongo_username}:{self.database.mongo_password}@{self.database.mongo_url.replace('mongodb://', '')}"
        return self.database.mongo_url
    
    def get_embedding_server_url(self) -> str:
        """Get complete embedding server URL"""
        if self.server.embedding_url.startswith('http'):
            return self.server.embedding_url
        return f"http://{self.server.embedding_host}:{self.server.embedding_port}/predict"


# Global configuration instance
config = RTCacheConfig()


def get_config() -> RTCacheConfig:
    """Get the global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment variables"""
    global config
    config = RTCacheConfig()


def update_config(**kwargs):
    """Update specific configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to find nested attribute
            for section_name in ['database', 'server', 'paths', 'model', 
                               'dataset', 'retrieval', 'experiment', 
                               'security', 'monitoring', 'development']:
                section = getattr(config, section_name)
                if hasattr(section, key):
                    setattr(section, key, value)
                    break


# Convenience functions for common configurations
def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.database


def get_server_config() -> ServerConfig:
    """Get server configuration"""
    return config.server


def get_paths_config() -> PathConfig:
    """Get paths configuration"""
    return config.paths


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.model


if __name__ == "__main__":
    # Example usage and testing
    config = get_config()
    
    print("RT-Cache Configuration:")
    print(f"Data Root: {config.paths.data_root}")
    print(f"Embedding Server: {config.get_embedding_server_url()}")
    print(f"MongoDB: {config.database.mongo_url}")
    print(f"Qdrant: {config.database.qdrant_host}:{config.database.qdrant_port}")
    print(f"Device: {config.model.device}")
    
    # Save example configuration
    config.save_to_yaml("./config/rt_cache_full_config.yaml")