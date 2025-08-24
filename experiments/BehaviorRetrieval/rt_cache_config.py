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
        "text_collection": {"dim": 4096, "distance": "Cosine"},   # OpenVLA text
        "clip_image_collection": {"dim": 512, "distance": "Cosine"},  # CLIP image
        "clip_text_collection": {"dim": 512, "distance": "Cosine"},   # CLIP text
        "byol_collection": {"dim": 2048, "distance": "Cosine"}        # BYOL embeddings
    }
    
    # Embedding server
    EMBEDDING_SERVER_URL = "http://localhost:8000/predict"
    
    # Dataset configuration
    DATASETS = [
        "berkeley_cable_routing", "roboturk", "nyu_door_opening_surprising_effectiveness", 
        "viola", "berkeley_autolab_ur5", "toto", "columbia_cairlab_pusht_real", 
        "austin_sailor_dataset_converted_externally_to_rlds", 
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
    ]
    
    # Data processing
    BATCH_INSERT_SIZE = 20
    IMAGE_SIZE = (224, 224)  # Standard input size
    
    # ============================================================================
    # VINN-Specific Settings
    # ============================================================================
    
    class VINN:
        # Online k-NN with cosine distance
        K_NEIGHBORS = 5
        DISTANCE_METRIC = "cosine"
        SEARCH_COLLECTION = "byol_collection"  # Use BYOL 2048-D embeddings
        
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
        EMBEDDING_SOURCE = "byol_collection"
        
    # ============================================================================
    # Evaluation Settings
    # ============================================================================
    
    class Evaluation:
        # Metrics
        METRICS = [
            "action_prediction_error",
            "inference_time", 
            "success_rate",
            "trajectory_smoothness"
        ]
        
        # Test settings
        NUM_TEST_EPISODES = 100
        NUM_RUNS_PER_TEST = 5
        EPISODE_LENGTH = 50
        
        # Success criteria
        POSITION_TOLERANCE = 0.05  # 5cm
        ROTATION_TOLERANCE = 0.1   # ~5.7 degrees
        
    # ============================================================================
    # Hardware/Training Settings
    # ============================================================================
    
    class Hardware:
        # Training efficiency: trains in < 5h on 1× A100
        MAX_TRAINING_TIME_HOURS = 5
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        NUM_WORKERS = 4
        
        # Memory optimization
        GRADIENT_CHECKPOINTING = True
        MIXED_PRECISION = True
        
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
            "image_size": cls.IMAGE_SIZE
        }
    
    @classmethod
    def get_behavior_retrieval_config(cls) -> Dict:
        """Get configuration dictionary for Behavior Retrieval"""
        return {
            "embedding_dim": cls.EMBEDDING_DIM,
            "action_dim": cls.ACTION_DIM,
            "vae_latent_dim": cls.BehaviorRetrieval.VAE_LATENT_DIM,
            "vae_input_dim": cls.BehaviorRetrieval.VAE_INPUT_DIM,
            "vae_hidden_dims": cls.BehaviorRetrieval.VAE_HIDDEN_DIMS,
            "vae_beta": cls.BehaviorRetrieval.VAE_BETA,
            "bc_hidden_dims": cls.BehaviorRetrieval.BC_HIDDEN_DIMS,
            "retrieval_fraction": cls.BehaviorRetrieval.RETRIEVAL_FRACTION,
            "collection_name": cls.BehaviorRetrieval.EMBEDDING_SOURCE,
            "qdrant_host": cls.QDRANT_HOST,
            "qdrant_port": cls.QDRANT_PORT,
            "mongo_url": cls.MONGODB_URL,
            "embedding_server": cls.EMBEDDING_SERVER_URL,
            "image_size": cls.IMAGE_SIZE,
            "training": {
                "vae_epochs": cls.BehaviorRetrieval.VAE_EPOCHS,
                "vae_batch_size": cls.BehaviorRetrieval.VAE_BATCH_SIZE,
                "vae_lr": cls.BehaviorRetrieval.VAE_LEARNING_RATE,
                "bc_epochs": cls.BehaviorRetrieval.BC_EPOCHS,
                "bc_batch_size": cls.BehaviorRetrieval.BC_BATCH_SIZE,
                "bc_lr": cls.BehaviorRetrieval.BC_LEARNING_RATE
            }
        }
    
    @classmethod
    def get_comparison_config(cls) -> Dict:
        """Get configuration for fair comparison"""
        return {
            "common_setup": {
                "vision_backbone": cls.VISION_BACKBONE,
                "representation": cls.REPRESENTATION,
                "embedding_dim": cls.EMBEDDING_DIM,
                "action_space": cls.ACTION_SPACE,
                "action_dim": cls.ACTION_DIM,
                "action_range": cls.ACTION_RANGE,
                "unlabeled_data": cls.UNLABELED_PRIOR_DATA,
                "demos_per_task": cls.DEMONSTRATIONS_PER_TASK
            },
            "evaluation": {
                "metrics": cls.Evaluation.METRICS,
                "num_test_episodes": cls.Evaluation.NUM_TEST_EPISODES,
                "num_runs": cls.Evaluation.NUM_RUNS_PER_TEST,
                "position_tolerance": cls.Evaluation.POSITION_TOLERANCE,
                "rotation_tolerance": cls.Evaluation.ROTATION_TOLERANCE
            },
            "constraints": {
                "no_additional_pretraining": True,
                "no_task_specific_finetuning": True,
                "same_backbone_frozen": True,
                "max_training_hours": cls.Hardware.MAX_TRAINING_TIME_HOURS
            }
        }
    
    @classmethod
    def validate_fair_comparison(cls, vinn_results: Dict, br_results: Dict) -> Dict:
        """
        Validate that fair comparison constraints are met
        
        Args:
            vinn_results: VINN evaluation results
            br_results: Behavior Retrieval evaluation results
            
        Returns:
            Validation results dictionary
        """
        validation = {
            "embedding_consistency": False,
            "action_space_consistency": False,
            "backbone_consistency": False,
            "data_consistency": False,
            "overall_valid": False
        }
        
        # Check embedding dimensions
        if (vinn_results.get("embedding_dim") == cls.EMBEDDING_DIM and 
            br_results.get("embedding_dim") == cls.EMBEDDING_DIM):
            validation["embedding_consistency"] = True
        
        # Check action space dimensions
        if (vinn_results.get("action_dim") == cls.ACTION_DIM and
            br_results.get("action_dim") == cls.ACTION_DIM):
            validation["action_space_consistency"] = True
        
        # Check backbone usage (both should use frozen ResNet-50)
        if (vinn_results.get("backbone") == cls.VISION_BACKBONE and
            br_results.get("backbone") == cls.VISION_BACKBONE):
            validation["backbone_consistency"] = True
        
        # Check data source (both should use RT-cache)
        if (vinn_results.get("data_source") == "RT-cache" and
            br_results.get("data_source") == "RT-cache"):
            validation["data_consistency"] = True
        
        # Overall validation
        validation["overall_valid"] = all([
            validation["embedding_consistency"],
            validation["action_space_consistency"], 
            validation["backbone_consistency"],
            validation["data_consistency"]
        ])
        
        return validation
    
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
        print("Behavior Retrieval Settings:")
        print(f"  - VAE Latent Dim: {cls.BehaviorRetrieval.VAE_LATENT_DIM}")
        print(f"  - Retrieval Fraction: {cls.BehaviorRetrieval.RETRIEVAL_FRACTION*100:.1f}%")
        print(f"  - Collection: {cls.BehaviorRetrieval.EMBEDDING_SOURCE}")
        print()
        print("Constraints:")
        print(f"  - No additional visual pre-training: ✓")
        print(f"  - No task-specific fine-tuning: ✓")
        print(f"  - Same frozen backbone: ✓")
        print(f"  - Training time limit: {cls.Hardware.MAX_TRAINING_TIME_HOURS}h on 1× A100")
        print("=" * 80)


# Make torch available for device detection
try:
    import torch
except ImportError:
    torch = None


def main():
    """
    Print configuration summary
    """
    RTCacheConfig.print_config_summary()
    
    # Print specific configs
    print("\nVINN Configuration:")
    vinn_config = RTCacheConfig.get_vinn_config()
    for key, value in vinn_config.items():
        print(f"  {key}: {value}")
    
    print("\nBehavior Retrieval Configuration:")
    br_config = RTCacheConfig.get_behavior_retrieval_config()
    for key, value in br_config.items():
        if key == "training":
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()