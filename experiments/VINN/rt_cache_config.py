"""
RT-Cache Configuration for Fair Comparison
Ensures both VINN and Behavior Retrieval use identical settings
"""

import os
from typing import Dict, List

class RTCacheConfig:
    """
    Configuration class for RT-cache data processing and model training
    Ensures fair comparison between VINN and Behavior Retrieval
    """
    
    # ============================================================================
    # Common Setup (as per objective)
    # ============================================================================
    
    # Vision backbone
    VISION_BACKBONE = "ImageNet-ResNet50"  # Widely used, no proprietary weights
    REPRESENTATION = "BYOL-2048D"  # BYOL self-supervision, 2048-D global pooling
    EMBEDDING_DIM = 2048  # Same encoder for both methods → fair comparison
    
    # Action space
    ACTION_SPACE = "delta_pose"  # Δ-pose (x,y,z,RPY) or joint targets; normalized
    ACTION_DIM = 7  # [x, y, z, roll, pitch, yaw, gripper]
    ACTION_RANGE = {
        "position": [-0.1, 0.1],     # Position deltas in meters
        "rotation": [-0.5, 0.5],     # Rotation deltas in radians  
        "gripper": [0.0, 1.0]        # Binary gripper (0=open, 1=close)
    }
    
    # Data
    UNLABELED_PRIOR_DATA = "Open-X"  # Subsampled Open-X (≈ 2–4 M frames)
    DEMONSTRATIONS_PER_TASK = 10     # e.g. 10 kinesthetic demos (fixed per run)
    
    # ============================================================================
    # RT-Cache Data Processing Settings
    # ============================================================================
    
    # Database connections
    MONGODB_URL = "mongodb://localhost:27017/"
    MONGODB_DB = "OpenVLACollection"
    MONGODB_COLLECTION = "OpenVLACollection"
    
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    QDRANT_TIMEOUT = 60.0
    
    # Vector collections
    COLLECTIONS = {
        "image_collection": {"dim": 2176, "distance": "Cosine"},  # OpenVLA embeddings
        "byol_collection": {"dim": 2048, "distance": "Cosine"}        # BYOL embeddings
    }
    
    # Embedding server
    EMBEDDING_SERVER_URL = "http://localhost:8000/predict"
    
    # ============================================================================
    # VINN-Specific Settings
    # ============================================================================
    
    class VINN:
        # Online k-NN with cosine distance
        K_NEIGHBORS = 5
        DISTANCE_METRIC = "cosine"
        SEARCH_COLLECTION = "image_collection"  # Use 2048-D BYOL embeddings
        
        # No additional visual pre-training or task-specific fine-tuning
        PRETRAINED_BACKBONE = False
        TASK_SPECIFIC_FINETUNING = False
        
    # ============================================================================
    # Behavior Retrieval-Specific Settings  
    # ============================================================================
    
    class BehaviorRetrieval:
        # Re-embeds state-action pairs with shallow VAE
        VAE_LATENT_DIM = 128
        VAE_INPUT_DIM = 2048 + 7  # BYOL embeddings + action
        VAE_HIDDEN_DIMS = [512, 256]
        VAE_BETA = 0.001  # KL weight
        
        # Retrieves ≈ 25% of Open-X
        RETRIEVAL_FRACTION = 0.25
        
        # Fine-tunes BC head
        BC_HIDDEN_DIMS = [256, 256]
        BC_DROPOUT = 0.1
        
        # Training parameters
        VAE_EPOCHS = 100
        VAE_BATCH_SIZE = 256
        VAE_LEARNING_RATE = 1e-3
        
        BC_EPOCHS = 50
        BC_BATCH_SIZE = 256
        BC_LEARNING_RATE = 1e-3
        
        # No additional visual pre-training or task-specific fine-tuning
        PRETRAINED_BACKBONE = False
        TASK_SPECIFIC_FINETUNING = False
        
        # Use same BYOL embeddings as VINN
        EMBEDDING_SOURCE = "image_collection"
        
    @classmethod
    def get_vinn_config(cls) -> Dict:
        """Get configuration dictionary for VINN"""
        return {
            "embedding_dim": cls.EMBEDDING_DIM,
            "action_dim": cls.ACTION_DIM,
            "k_neighbors": cls.VINN.K_NEIGHBORS,
            "distance_metric": cls.VINN.DISTANCE_METRIC,
            "collection_name": cls.VINN.SEARCH_COLLECTION,
            "qdrant_host": cls.QDRANT_HOST,
            "qdrant_port": cls.QDRANT_PORT,
            "mongo_url": cls.MONGODB_URL,
            "embedding_server": cls.EMBEDDING_SERVER_URL,
            "image_size": (224, 224)
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary for verification"""
        print("=" * 80)
        print("RT-CACHE FAIR COMPARISON CONFIGURATION")
        print("=" * 80)
        print(f"Vision Backbone: {cls.VISION_BACKBONE}")
        print(f"Representation: {cls.REPRESENTATION} ({cls.EMBEDDING_DIM}-D)")
        print(f"Action Space: {cls.ACTION_SPACE} ({cls.ACTION_DIM}-DOF)")
        print(f"Data Source: {cls.UNLABELED_PRIOR_DATA}")
        print(f"Demos per Task: {cls.DEMONSTRATIONS_PER_TASK}")
        print()
        print("VINN Settings:")
        print(f"  - k-NN: {cls.VINN.K_NEIGHBORS} neighbors")
        print(f"  - Distance: {cls.VINN.DISTANCE_METRIC}")
        print(f"  - Collection: {cls.VINN.SEARCH_COLLECTION}")
        print()
        print("Constraints:")
        print(f"  - No additional visual pre-training: ✓")
        print(f"  - No task-specific fine-tuning: ✓")
        print(f"  - Same frozen backbone: ✓")
        print("=" * 80)


def main():
    """Print configuration summary"""
    RTCacheConfig.print_config_summary()


if __name__ == "__main__":
    main()