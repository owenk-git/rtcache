#!/usr/bin/env python3
"""
Dataset Processing Script for RT-Cache System

This script processes robot demonstration data from various Open X-Embodiment datasets,
extracts actions and observations, generates embeddings, and stores them in MongoDB
and Qdrant vector database for efficient retrieval.

Author: RT-Cache Team
Date: 2024
"""

import os
import sys
import argparse
import logging
import time
import uuid
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import yaml
import requests
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import get_config

# Note: These imports are not available in the current structure
# Users should implement these classes or use direct implementations

def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Load environment variables and config
load_dotenv()
config = get_config()

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for dataset processing"""
    
    # Paths
    data_root: str = config.paths.data_root
    image_dir: str = config.paths.image_storage_path
    
    # Server URLs
    embedding_server_url: str = config.server.embedding_url
    
    # Database settings
    mongo_url: str = config.database.mongo_url
    mongo_db: str = config.database.mongo_db_name
    qdrant_host: str = config.database.qdrant_host
    qdrant_port: int = config.database.qdrant_port
    
    # Processing parameters
    batch_size: int = config.dataset.processing_batch_size
    max_episodes: int = config.dataset.max_episodes
    min_episode_length: int = config.dataset.min_episode_length
    
    # Embedding dimensions
    openvla_image_dim: int = config.retrieval.openvla_image_dim
    openvla_text_dim: int = config.retrieval.openvla_text_dim
    clip_image_dim: int = config.retrieval.clip_image_dim
    clip_text_dim: int = config.retrieval.clip_text_dim

# ============================================================================
# Dataset Processing
# ============================================================================

class DatasetProcessor:
    """
    Main class for processing robot demonstration datasets.
    
    This class handles:
    - Loading datasets from TensorFlow Datasets
    - Extracting observations and actions
    - Generating embeddings via remote server
    - Storing data in MongoDB and Qdrant
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the dataset processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage backends
        self._init_storage()
        
        # Initialize GPU settings
        self._init_gpu()
        
        # Load dataset configurations
        self._load_dataset_configs()
        
        # Statistics tracking
        self.stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "skipped_episodes": 0,
            "failed_embeddings": 0
        }
        
    def _init_gpu(self):
        """Initialize GPU memory settings for TensorFlow"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Initialized {len(gpus)} GPU(s) with memory growth enabled")
            except RuntimeError as e:
                self.logger.error(f"GPU initialization error: {e}")
                
    def _init_storage(self):
        """Initialize MongoDB and Qdrant connections"""
        # MongoDB
        self.mongo_client = MongoClient(self.config.mongo_url)
        self.mongo_db = self.mongo_client[self.config.mongo_db]
        self.mongo_collection = self.mongo_db["trajectories"]
        
        # Qdrant
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            timeout=60.0
        )
        
        # Create Qdrant collections if they don't exist
        self._create_qdrant_collections()
        
    def _create_qdrant_collections(self):
        """Create Qdrant vector collections"""
        collections = [
            ("image_collection", self.config.openvla_image_dim),
            ("text_collection", self.config.openvla_text_dim),
            ("clip_image_collection", self.config.clip_image_dim),
            ("clip_text_collection", self.config.clip_text_dim)
        ]
        
        for name, dim in collections:
            if not self.qdrant_client.collection_exists(name):
                self.qdrant_client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance="Cosine")
                )
                self.logger.info(f"Created Qdrant collection: {name} (dim={dim})")
                
    def _load_dataset_configs(self):
        """Load dataset configurations from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "datasets.yaml"
        with open(config_path, 'r') as f:
            self.dataset_configs = yaml.safe_load(f)['datasets']
            
    def process_datasets(self, dataset_names: List[str]):
        """
        Process multiple datasets.
        
        Args:
            dataset_names: List of dataset names to process
        """
        for dataset_name in dataset_names:
            if dataset_name not in self.dataset_configs:
                self.logger.warning(f"Unknown dataset: {dataset_name}, skipping")
                continue
                
            self.logger.info(f"Processing dataset: {dataset_name}")
            start_time = time.time()
            
            try:
                self.process_single_dataset(dataset_name)
            except Exception as e:
                self.logger.error(f"Failed to process {dataset_name}: {e}")
                continue
                
            elapsed = time.time() - start_time
            self.logger.info(f"Completed {dataset_name} in {elapsed:.2f} seconds")
            
        # Print final statistics
        self._print_statistics()
        
    def process_single_dataset(self, dataset_name: str):
        """
        Process a single dataset.
        
        Args:
            dataset_name: Name of the dataset to process
        """
        # Load dataset
        dataset_path = self._get_dataset_path(dataset_name)
        builder = tfds.builder_from_directory(builder_dir=dataset_path)
        ds = builder.as_dataset(split='train', shuffle_files=False)
        
        # Determine dataset structure (RLDS or standard)
        is_rlds = 'steps' in builder.info.features
        
        # Find observation keys
        obs_keys = self._get_observation_keys(builder, is_rlds)
        
        if not obs_keys['image']:
            self.logger.warning(f"No valid image key found in {dataset_name}")
            return
            
        # Process episodes
        if is_rlds:
            self._process_rlds_dataset(ds, dataset_name, obs_keys)
        else:
            self._process_standard_dataset(ds, dataset_name, obs_keys)
            
    def _process_rlds_dataset(self, dataset, dataset_name: str, obs_keys: Dict):
        """
        Process RLDS (Reinforcement Learning Dataset) format.
        
        Args:
            dataset: TensorFlow dataset
            dataset_name: Name of the dataset
            obs_keys: Dictionary of observation keys
        """
        batch_buffer = {
            'mongo_docs': [],
            'image_points': [],
            'text_points': [],
            'clip_image_points': [],
            'clip_text_points': []
        }
        
        for episode_idx, episode in enumerate(dataset):
            if episode_idx >= self.config.max_episodes:
                break
                
            # Convert steps to list for easier processing
            steps_dataset = episode['steps']
            steps_list = list(steps_dataset.as_numpy_iterator())
            
            if len(steps_list) < self.config.min_episode_length:
                self.stats['skipped_episodes'] += 1
                continue
                
            self.stats['total_episodes'] += 1
            
            # Extract episode text if available
            episode_text = self._extract_episode_text(steps_list[0], obs_keys)
            
            # Process each step
            for step_idx, step in enumerate(steps_list):
                self.stats['total_steps'] += 1
                
                # Process step data
                step_data = self._process_step(
                    step=step,
                    dataset_name=dataset_name,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    episode_text=episode_text,
                    obs_keys=obs_keys,
                    total_steps=len(steps_list)
                )
                
                if step_data:
                    self._add_to_batch(batch_buffer, step_data)
                    
                # Flush batch if needed
                if len(batch_buffer['mongo_docs']) >= self.config.batch_size:
                    self._flush_batch(batch_buffer)
                    
            # Process episode text embedding
            if episode_text:
                text_data = self._process_text_embedding(
                    text=episode_text,
                    dataset_name=dataset_name,
                    episode_idx=episode_idx,
                    total_steps=len(steps_list)
                )
                if text_data:
                    self._add_text_to_batch(batch_buffer, text_data)
                    
        # Final flush
        self._flush_batch(batch_buffer)
        
    def _process_step(self, step: Dict, dataset_name: str, episode_idx: int,
                     step_idx: int, episode_text: str, obs_keys: Dict,
                     total_steps: int) -> Optional[Dict]:
        """
        Process a single step from the dataset.
        
        Args:
            step: Step data dictionary
            dataset_name: Name of the dataset
            episode_idx: Episode index
            step_idx: Step index within episode
            episode_text: Text instruction for the episode
            obs_keys: Observation keys
            total_steps: Total steps in episode
            
        Returns:
            Processed step data or None if processing failed
        """
        try:
            # Extract observation and action
            obs = step['observation']
            action = step['action']
            
            # Extract and normalize action
            action_vector = self._extract_action(action)
            normalized_action = self._normalize_action(action_vector)
            
            # Create unique ID
            doc_id = f"{dataset_name}_{episode_idx}_{step_idx}"
            
            # Process and save image
            image_path = self._save_image(obs[obs_keys['image']], doc_id)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(image_path, episode_text)
            
            if not embeddings:
                self.stats['failed_embeddings'] += 1
                return None
                
            return {
                'doc_id': doc_id,
                'dataset_name': dataset_name,
                'episode_idx': episode_idx,
                'step_idx': step_idx,
                'total_steps': total_steps,
                'raw_action': action_vector.tolist(),
                'normalized_action': normalized_action.tolist(),
                'text': episode_text,
                'embeddings': embeddings,
                'image_path': str(image_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing step: {e}")
            return None
            
    def _extract_action(self, action_data) -> np.ndarray:
        """
        Extract action vector from various formats.
        
        Args:
            action_data: Action data in various formats
            
        Returns:
            7-dimensional action vector [x, y, z, roll, pitch, yaw, gripper]
        """
        if isinstance(action_data, dict):
            # Extract components
            world_vector = self._ensure_rank1(
                action_data.get('world_vector'), (3,)
            )
            rotation_delta = self._ensure_rank1(
                action_data.get('rotation_delta'), (3,)
            )
            gripper = self._ensure_rank1(
                action_data.get('gripper_closedness_action'), (1,)
            )
            
            # Combine into single vector
            combined = tf.concat([world_vector, rotation_delta, gripper], axis=-1)
            return combined.numpy().astype(np.float32)
            
        elif isinstance(action_data, tf.Tensor) and action_data.shape[-1] == 7:
            return action_data.numpy().astype(np.float32)
            
        else:
            return np.zeros(7, dtype=np.float32)
            
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action vector to standard ranges.
        
        Args:
            action: Raw action vector
            
        Returns:
            Normalized action vector
        """
        normalized = action.copy()
        
        # Position: clip to [-0.1, 0.1]
        normalized[:3] = np.clip(normalized[:3], -0.1, 0.1)
        
        # Rotation: clip to [-0.5, 0.5]
        normalized[3:6] = np.clip(normalized[3:6], -0.5, 0.5)
        
        # Gripper: binary (0 or 1)
        normalized[6] = 1.0 if normalized[6] > 0 else 0.0
        
        return normalized
        
    def _save_image(self, image_data, doc_id: str) -> Path:
        """
        Save image to disk.
        
        Args:
            image_data: Image data (numpy array or tensor)
            doc_id: Document ID for naming
            
        Returns:
            Path to saved image
        """
        # Ensure image directory exists
        image_dir = Path(self.config.image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy if needed
        if isinstance(image_data, tf.Tensor):
            image_data = image_data.numpy()
            
        # Convert to uint8
        if image_data.dtype != np.uint8:
            image_data = (image_data * 255).astype(np.uint8)
            
        # Save image
        image = Image.fromarray(image_data)
        image_path = image_dir / f"{doc_id}.png"
        
        if not image_path.exists():
            image.save(image_path)
            
        return image_path
        
    def _generate_embeddings(self, image_path: Path, text: Optional[str]) -> Dict:
        """
        Generate embeddings via remote server.
        
        Args:
            image_path: Path to image file
            text: Optional text instruction
            
        Returns:
            Dictionary of embeddings
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Prepare request
            files = {}
            buf = BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            files["file"] = ("image.png", buf, "image/png")
            
            data = {
                "instruction": text if text else "",
                "option": "image"  # Get image embeddings only
            }
            
            # Send request
            response = requests.post(
                self.config.embedding_server_url,
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            # Decode embeddings
            result = response.json()
            embeddings = {}
            
            # OpenVLA image features
            if "image_features" in result:
                embeddings["openvla_image"] = self._decode_base64_tensor(
                    result["image_features"]
                )
                
            # CLIP image features
            if "clip_image_features" in result:
                embeddings["clip_image"] = self._decode_base64_tensor(
                    result["clip_image_features"]
                )
                
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return {}
            
    def _decode_base64_tensor(self, b64_string: str) -> List[float]:
        """
        Decode base64-encoded tensor.
        
        Args:
            b64_string: Base64 encoded tensor
            
        Returns:
            List of float values
        """
        binary_data = base64.b64decode(b64_string)
        buffer = BytesIO(binary_data)
        tensor = torch.load(buffer, map_location="cpu")
        return tensor.squeeze(0).tolist()
        
    def _flush_batch(self, batch_buffer: Dict):
        """
        Flush batch buffer to databases.
        
        Args:
            batch_buffer: Dictionary containing buffered data
        """
        # Insert to MongoDB
        if batch_buffer['mongo_docs']:
            self.mongo_collection.insert_many(batch_buffer['mongo_docs'])
            batch_buffer['mongo_docs'].clear()
            
        # Insert to Qdrant collections
        collections = [
            ('image_points', 'image_collection'),
            ('clip_image_points', 'clip_image_collection'),
            ('text_points', 'text_collection'),
            ('clip_text_points', 'clip_text_collection')
        ]
        
        for buffer_key, collection_name in collections:
            if batch_buffer[buffer_key]:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch_buffer[buffer_key]
                )
                batch_buffer[buffer_key].clear()
                
    def _print_statistics(self):
        """Print processing statistics"""
        self.logger.info("=" * 60)
        self.logger.info("Processing Statistics:")
        self.logger.info(f"  Total episodes: {self.stats['total_episodes']}")
        self.logger.info(f"  Total steps: {self.stats['total_steps']}")
        self.logger.info(f"  Skipped episodes: {self.stats['skipped_episodes']}")
        self.logger.info(f"  Failed embeddings: {self.stats['failed_embeddings']}")
        self.logger.info("=" * 60)
        
    # Helper methods
    def _ensure_rank1(self, value, default_shape):
        """Ensure tensor has rank 1"""
        t = tf.convert_to_tensor(value) if value is not None else tf.zeros(default_shape)
        if t.shape.ndims == 0:
            t = tf.expand_dims(t, 0)
        return t
        
    def _get_dataset_path(self, dataset_name: str) -> str:
        """Get dataset path from name"""
        # Version mapping (adjust as needed)
        if dataset_name == 'robo_net':
            version = '1.0.0'
        elif dataset_name == 'language_table':
            version = '0.0.1'
        else:
            version = '0.1.0'
        return f'gs://gresearch/robotics/{dataset_name}/{version}'
        
    def _get_observation_keys(self, builder, is_rlds: bool) -> Dict:
        """Extract observation keys from dataset"""
        if is_rlds:
            obs_keys = list(builder.info.features['steps']['observation'].keys())
        else:
            obs_keys = list(builder.info.features.keys())
            
        # Find image key
        possible_image_keys = [
            'image', 'rgb_static', 'front_rgb', 'agentview_rgb',
            'rgb', 'hand_image', 'image_1'
        ]
        image_key = next((k for k in possible_image_keys if k in obs_keys), None)
        
        # Find text key
        possible_text_keys = ['natural_language_instruction', 'language_instruction']
        text_key = next((k for k in possible_text_keys if k in obs_keys), None)
        
        return {'image': image_key, 'text': text_key}
        
    def _extract_episode_text(self, first_step: Dict, obs_keys: Dict) -> Optional[str]:
        """Extract episode text from first step"""
        if obs_keys['text'] and obs_keys['text'] in first_step['observation']:
            text_data = first_step['observation'][obs_keys['text']]
            if isinstance(text_data, bytes):
                return text_data.decode('utf-8')
            elif isinstance(text_data, str):
                return text_data
        return None
        
    def _process_standard_dataset(self, dataset, dataset_name: str, obs_keys: Dict):
        """Process standard (non-RLDS) dataset format"""
        # Implementation similar to _process_rlds_dataset but for standard format
        # This would handle datasets that don't follow the RLDS structure
        pass
        
    def _process_text_embedding(self, text: str, dataset_name: str,
                               episode_idx: int, total_steps: int) -> Optional[Dict]:
        """Generate text embeddings"""
        # Implementation for text embedding generation
        pass
        
    def _add_to_batch(self, batch_buffer: Dict, step_data: Dict):
        """Add step data to batch buffer"""
        # Create MongoDB document
        mongo_doc = {
            'id': step_data['doc_id'],
            'dataset_name': step_data['dataset_name'],
            'episode_idx': step_data['episode_idx'],
            'step_idx': step_data['step_idx'],
            'total_steps': step_data['total_steps'],
            'raw_action': step_data['raw_action'],
            'normalized_action': step_data['normalized_action'],
            'text': step_data['text'],
            'image_path': step_data['image_path']
        }
        batch_buffer['mongo_docs'].append(mongo_doc)
        
        # Create Qdrant points
        if 'openvla_image' in step_data['embeddings']:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=step_data['embeddings']['openvla_image'],
                payload={
                    'logical_id': step_data['doc_id'],
                    'dataset_name': step_data['dataset_name'],
                    'episode_idx': step_data['episode_idx'],
                    'step_idx': step_data['step_idx'],
                    'text': step_data['text']
                }
            )
            batch_buffer['image_points'].append(point)
            
        if 'clip_image' in step_data['embeddings']:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=step_data['embeddings']['clip_image'],
                payload={
                    'logical_id': step_data['doc_id'],
                    'dataset_name': step_data['dataset_name'],
                    'episode_idx': step_data['episode_idx'],
                    'step_idx': step_data['step_idx'],
                    'text': step_data['text']
                }
            )
            batch_buffer['clip_image_points'].append(point)
            
    def _add_text_to_batch(self, batch_buffer: Dict, text_data: Dict):
        """Add text embedding data to batch buffer"""
        # Implementation for adding text embeddings to batch
        pass

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the script"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process robot demonstration datasets for RT-Cache"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets to process (or 'all')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=100,
        help="Maximum episodes per dataset"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = ProcessingConfig(
        batch_size=args.batch_size,
        max_episodes=args.max_episodes
    )
    
    # Get dataset list
    if args.datasets.lower() == 'all':
        # Load all datasets from config
        config_path = Path(__file__).parent.parent.parent / "config" / "datasets.yaml"
        with open(config_path, 'r') as f:
            dataset_names = list(yaml.safe_load(f)['datasets'].keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(',')]
        
    logger.info(f"Processing {len(dataset_names)} datasets")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create processor and run
    processor = DatasetProcessor(config)
    processor.process_datasets(dataset_names)
    
    logger.info("Dataset processing complete!")

if __name__ == "__main__":
    main()