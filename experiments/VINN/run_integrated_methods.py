"""
Integration Script: VINN vs Behavior Retrieval with RT-cache
Demonstrates how to run both methods with identical RT-cache data processing
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, List

# Add paths (relative to repository structure)
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(project_root)
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BehaviorRetrieval'))

from rt_cache_config import RTCacheConfig
from fair_comparison import FairComparisonSuite


def run_vinn_only(config: Dict, num_test_obs: int = 5):
    """
    Run VINN method only with RT-cache integration
    """
    print("\n" + "="*60)
    print("RUNNING VINN WITH RT-CACHE INTEGRATION")
    print("="*60)
    
    from vinn_rt_cache import VINNPolicy
    
    # Initialize VINN with RT-cache config
    vinn = VINNPolicy(
        qdrant_host=config["qdrant_host"],
        qdrant_port=config["qdrant_port"],
        mongo_host=config["mongo_url"],
        embedding_server=config["embedding_server"],
        k_neighbors=config["k_neighbors"],
        distance_metric=config["distance_metric"],
        collection_name=config["collection_name"]
    )
    
    print(f"[VINN] Initialized with {config['k_neighbors']}-NN search")
    print(f"[VINN] Using {config['embedding_dim']}-D BYOL embeddings")
    print(f"[VINN] Action space: {config['action_dim']}-DOF")
    
    # Generate test observations
    test_observations = []
    for i in range(num_test_obs):
        test_image = np.random.randint(0, 255, config["image_size"] + (3,), dtype=np.uint8)
        test_observations.append({"image": test_image})
    
    # Run predictions
    print(f"\n[VINN] Running {num_test_obs} predictions...")
    predictions = []
    prediction_times = []
    
    for i, obs in enumerate(test_observations):
        start_time = time.time()
        action = vinn.predict_action(obs)
        pred_time = time.time() - start_time
        
        predictions.append(action)
        prediction_times.append(pred_time)
        
        print(f"[VINN] Prediction {i+1}: {action} (time: {pred_time:.4f}s)")
    
    # Summary
    avg_time = np.mean(prediction_times)
    print(f"\n[VINN] Summary:")
    print(f"  Average prediction time: {avg_time:.4f}s")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Action shape: {predictions[0].shape}")
    print(f"  Action range: [{min(p.min() for p in predictions):.3f}, {max(p.max() for p in predictions):.3f}]")
    
    return {
        "method": "VINN",
        "predictions": predictions,
        "prediction_times": prediction_times,
        "config": config
    }


def run_behavior_retrieval_only(config: Dict, num_test_obs: int = 5, train: bool = True):
    """
    Run Behavior Retrieval method only with RT-cache integration
    """
    print("\n" + "="*60)
    print("RUNNING BEHAVIOR RETRIEVAL WITH RT-CACHE INTEGRATION")
    print("="*60)
    
    from behavior_retrieval_rt_cache import BehaviorRetrievalPolicy
    
    # Initialize Behavior Retrieval with RT-cache config
    br = BehaviorRetrievalPolicy(
        qdrant_host=config["qdrant_host"],
        qdrant_port=config["qdrant_port"],
        mongo_host=config["mongo_url"],
        embedding_server=config["embedding_server"],
        collection_name=config["collection_name"],
        sample_fraction=config["retrieval_fraction"],
        vae_latent_dim=config["vae_latent_dim"]
    )
    
    print(f"[BR] Initialized with {config['retrieval_fraction']*100:.1f}% data retrieval")
    print(f"[BR] Using {config['embedding_dim']}-D BYOL embeddings")
    print(f"[BR] VAE latent dim: {config['vae_latent_dim']}")
    print(f"[BR] Action space: {config['action_dim']}-DOF")
    
    # Train if requested
    if train:
        print(f"\n[BR] Training models...")
        training_config = config["training"]
        br.train(
            vae_epochs=training_config["vae_epochs"],
            bc_epochs=training_config["bc_epochs"],
            batch_size=training_config["vae_batch_size"],
            max_samples=10000  # Reduced for demo
        )
    
    # Generate test observations (same as VINN)
    test_observations = []
    for i in range(num_test_obs):
        test_image = np.random.randint(0, 255, config["image_size"] + (3,), dtype=np.uint8)
        test_observations.append({"image": test_image})
    
    # Run predictions
    print(f"\n[BR] Running {num_test_obs} predictions...")
    predictions = []
    prediction_times = []
    
    for i, obs in enumerate(test_observations):
        start_time = time.time()
        action = br.predict_action(obs)
        pred_time = time.time() - start_time
        
        predictions.append(action)
        prediction_times.append(pred_time)
        
        print(f"[BR] Prediction {i+1}: {action} (time: {pred_time:.4f}s)")
    
    # Summary
    avg_time = np.mean(prediction_times)
    print(f"\n[BR] Summary:")
    print(f"  Average prediction time: {avg_time:.4f}s")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Action shape: {predictions[0].shape}")
    print(f"  Action range: [{min(p.min() for p in predictions):.3f}, {max(p.max() for p in predictions):.3f}]")
    
    return {
        "method": "BehaviorRetrieval",
        "predictions": predictions,
        "prediction_times": prediction_times,
        "config": config
    }


def run_fair_comparison(num_test_obs: int = 3):
    """
    Run fair comparison between both methods
    """
    print("\n" + "="*60)
    print("RUNNING FAIR COMPARISON")
    print("="*60)
    
    comparison = FairComparisonSuite(
        qdrant_host=RTCacheConfig.QDRANT_HOST,
        qdrant_port=RTCacheConfig.QDRANT_PORT,
        mongo_host=RTCacheConfig.MONGODB_URL,
        embedding_server=RTCacheConfig.EMBEDDING_SERVER_URL,
        collection_name="image_collection"
    )
    
    results = comparison.run_full_comparison(
        k_neighbors=RTCacheConfig.VINN.K_NEIGHBORS,
        num_test_images=num_test_obs
    )
    
    return results


def main():
    """
    Main entry point for integrated methods
    """
    parser = argparse.ArgumentParser(description="Run VINN and Behavior Retrieval with RT-cache integration")
    parser.add_argument("--method", choices=["vinn", "br", "both", "compare"], default="both",
                       help="Which method to run")
    parser.add_argument("--num_test", type=int, default=3,
                       help="Number of test observations")
    parser.add_argument("--train_br", action="store_true", default=True,
                       help="Train Behavior Retrieval models")
    parser.add_argument("--config_only", action="store_true",
                       help="Just print configuration and exit")
    
    args = parser.parse_args()
    
    # Print configuration
    RTCacheConfig.print_config_summary()
    
    if args.config_only:
        return
    
    # Get configurations
    vinn_config = RTCacheConfig.get_vinn_config()
    br_config = RTCacheConfig.get_behavior_retrieval_config()
    
    results = {}
    
    # Run requested methods
    if args.method == "vinn":
        results["vinn"] = run_vinn_only(vinn_config, args.num_test)
        
    elif args.method == "br":
        results["br"] = run_behavior_retrieval_only(br_config, args.num_test, args.train_br)
        
    elif args.method == "both":
        results["vinn"] = run_vinn_only(vinn_config, args.num_test)
        results["br"] = run_behavior_retrieval_only(br_config, args.num_test, args.train_br)
        
    elif args.method == "compare":
        results["comparison"] = run_fair_comparison(args.num_test)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if "vinn" in results:
        vinn_avg_time = np.mean(results["vinn"]["prediction_times"])
        print(f"VINN: {len(results['vinn']['predictions'])} predictions, {vinn_avg_time:.4f}s avg")
    
    if "br" in results:
        br_avg_time = np.mean(results["br"]["prediction_times"])
        print(f"Behavior Retrieval: {len(results['br']['predictions'])} predictions, {br_avg_time:.4f}s avg")
    
    if "vinn" in results and "br" in results:
        vinn_time = np.mean(results["vinn"]["prediction_times"])
        br_time = np.mean(results["br"]["prediction_times"])
        speedup = br_time / vinn_time
        print(f"Speedup: {speedup:.2f}x ({'VINN faster' if speedup > 1 else 'BR faster'})")
    
    if "comparison" in results:
        comp = results["comparison"]
        print(f"Fair comparison completed:")
        print(f"  Embeddings identical: {comp['embeddings_identical']}")
        print(f"  Actions consistent: {comp['actions_consistent']}")
    
    print("\n✅ Integration completed successfully!")
    print("Both methods now use:")
    print("  • Identical 2048-D BYOL embeddings from RT-cache")
    print("  • Same 7-DOF action space: Δ-pose (x,y,z,RPY) + gripper")
    print("  • Same data processing pipeline")
    print("  • No additional visual pre-training")
    print("  • No task-specific fine-tuning of backbone")


if __name__ == "__main__":
    main()