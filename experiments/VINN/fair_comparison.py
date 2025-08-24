"""
Fair Comparison Script for VINN vs Behavior Retrieval
Ensures identical setup as per objective:
- Same 2048-D BYOL embeddings from frozen ResNet-50
- Same action space: Δ-pose (x,y,z,RPY) + gripper  
- Same data processing pipeline from RT-cache
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image

# Add paths for imports (relative to repository structure)
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BehaviorRetrieval'))

from vinn_rt_cache import VINNPolicy, VINNEvaluator
from behavior_retrieval_rt_cache import BehaviorRetrievalPolicy


class FairComparisonSuite:
    """
    Fair comparison suite ensuring identical conditions for VINN and Behavior Retrieval
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        mongo_host: str = "mongodb://localhost:27017/",
        embedding_server: str = "http://localhost:8000/predict",
        collection_name: str = "image_collection"
    ):
        """
        Initialize comparison suite with shared infrastructure
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.mongo_host = mongo_host
        self.embedding_server = embedding_server
        self.collection_name = collection_name
        
        # Initialize policies
        self.vinn_policy = None
        self.br_policy = None
        
        print("[FairComparison] Initialized comparison suite")
        print(f"[FairComparison] Using collection: {collection_name}")
    
    def setup_vinn(self, k_neighbors: int = 5):
        """
        Setup VINN policy with fair comparison settings
        """
        self.vinn_policy = VINNPolicy(
            qdrant_host=self.qdrant_host,
            qdrant_port=self.qdrant_port,
            mongo_host=self.mongo_host,
            embedding_server=self.embedding_server,
            k_neighbors=k_neighbors,
            distance_metric="cosine",
            collection_name=self.collection_name
        )
        print("[FairComparison] VINN policy setup complete")
    
    def setup_behavior_retrieval(self, train: bool = True):
        """
        Setup Behavior Retrieval policy with fair comparison settings
        """
        self.br_policy = BehaviorRetrievalPolicy(
            qdrant_host=self.qdrant_host,
            qdrant_port=self.qdrant_port,
            mongo_host=self.mongo_host,
            embedding_server=self.embedding_server,
            collection_name=self.collection_name,
            sample_fraction=0.25,  # Retrieve ~25% of Open-X
            vae_latent_dim=128
        )
        
        if train:
            print("[FairComparison] Training Behavior Retrieval...")
            self.br_policy.train(
                vae_epochs=50,
                bc_epochs=25,
                batch_size=128,
                max_samples=10000
            )
        
        print("[FairComparison] Behavior Retrieval policy setup complete")
    
    def verify_identical_embeddings(self, test_images: List[np.ndarray]) -> bool:
        """
        Verify both methods use identical 2048-D BYOL embeddings
        
        Args:
            test_images: List of test images to check
            
        Returns:
            True if embeddings are identical
        """
        print("[FairComparison] Verifying identical BYOL embeddings...")
        
        if not self.vinn_policy or not self.br_policy:
            raise ValueError("Both policies must be setup first")
        
        all_identical = True
        tolerance = 1e-6
        
        for i, image in enumerate(test_images):
            # Get BYOL embeddings from both methods
            vinn_embedding = self.vinn_policy._get_byol_embedding(image)
            br_embedding = self.br_policy._get_byol_embedding(image)
            
            # Check dimensions
            if vinn_embedding.shape[0] != 2048 or br_embedding.shape[0] != 2048:
                print(f"[ERROR] Embedding dimension mismatch: VINN={vinn_embedding.shape[0]}, BR={br_embedding.shape[0]}")
                all_identical = False
                continue
            
            # Check if embeddings are identical (within tolerance)
            diff = np.linalg.norm(vinn_embedding - br_embedding)
            if diff > tolerance:
                print(f"[ERROR] Embedding difference for image {i}: {diff}")
                all_identical = False
            else:
                print(f"[OK] Image {i}: Embeddings identical (diff={diff:.2e})")
        
        if all_identical:
            print("[SUCCESS] All BYOL embeddings are identical between VINN and BR")
        else:
            print("[FAILURE] Embedding differences detected")
        
        return all_identical
    
    def verify_action_space_consistency(self, test_observations: List[Dict]) -> bool:
        """
        Verify both methods use identical 7-DOF action space
        
        Args:
            test_observations: List of test observations
            
        Returns:
            True if action spaces are consistent
        """
        print("[FairComparison] Verifying action space consistency...")
        
        if not self.vinn_policy or not self.br_policy:
            raise ValueError("Both policies must be setup first")
        
        all_consistent = True
        
        for i, obs in enumerate(test_observations):
            try:
                # Get actions from both methods
                vinn_action = self.vinn_policy.predict_action(obs)
                br_action = self.br_policy.predict_action(obs)
                
                # Check dimensions (should be 7-DOF: x,y,z,roll,pitch,yaw,gripper)
                if vinn_action.shape[0] != 7:
                    print(f"[ERROR] VINN action dimension: {vinn_action.shape[0]} (expected 7)")
                    all_consistent = False
                
                if br_action.shape[0] != 7:
                    print(f"[ERROR] BR action dimension: {br_action.shape[0]} (expected 7)")
                    all_consistent = False
                
                # Check action ranges (normalized between -1 and 1 typically)
                vinn_range = (vinn_action.min(), vinn_action.max())
                br_range = (br_action.min(), br_action.max())
                
                print(f"[INFO] Obs {i}: VINN action range [{vinn_range[0]:.3f}, {vinn_range[1]:.3f}]")
                print(f"[INFO] Obs {i}: BR action range [{br_range[0]:.3f}, {br_range[1]:.3f}]")
                
                # Verify action components
                print(f"[INFO] Obs {i}: VINN action = {vinn_action}")
                print(f"[INFO] Obs {i}: BR action = {br_action}")
                
            except Exception as e:
                print(f"[ERROR] Failed to get actions for obs {i}: {e}")
                all_consistent = False
        
        if all_consistent:
            print("[SUCCESS] Action space consistency verified")
        else:
            print("[FAILURE] Action space inconsistencies detected")
        
        return all_consistent
    
    def run_performance_comparison(
        self, 
        test_observations: List[Dict], 
        num_runs: int = 5
    ) -> Dict:
        """
        Run performance comparison between VINN and Behavior Retrieval
        
        Args:
            test_observations: List of test observations
            num_runs: Number of runs for averaging
            
        Returns:
            Performance comparison results
        """
        print(f"[FairComparison] Running performance comparison ({num_runs} runs)...")
        
        if not self.vinn_policy or not self.br_policy:
            raise ValueError("Both policies must be setup first")
        
        vinn_times = []
        br_times = []
        vinn_actions = []
        br_actions = []
        
        for run in range(num_runs):
            print(f"[FairComparison] Run {run + 1}/{num_runs}")
            
            run_vinn_times = []
            run_br_times = []
            run_vinn_actions = []
            run_br_actions = []
            
            for obs in test_observations:
                # Time VINN prediction
                start_time = time.time()
                vinn_action = self.vinn_policy.predict_action(obs)
                vinn_time = time.time() - start_time
                
                # Time BR prediction  
                start_time = time.time()
                br_action = self.br_policy.predict_action(obs)
                br_time = time.time() - start_time
                
                run_vinn_times.append(vinn_time)
                run_br_times.append(br_time)
                run_vinn_actions.append(vinn_action)
                run_br_actions.append(br_action)
            
            vinn_times.extend(run_vinn_times)
            br_times.extend(run_br_times)
            vinn_actions.extend(run_vinn_actions)
            br_actions.extend(run_br_actions)
        
        # Compute statistics
        results = {
            "vinn": {
                "avg_time": np.mean(vinn_times),
                "std_time": np.std(vinn_times),
                "total_time": np.sum(vinn_times),
                "actions": vinn_actions
            },
            "behavior_retrieval": {
                "avg_time": np.mean(br_times),
                "std_time": np.std(br_times),
                "total_time": np.sum(br_times),
                "actions": br_actions
            },
            "comparison": {
                "speedup_factor": np.mean(br_times) / np.mean(vinn_times),
                "num_observations": len(test_observations),
                "num_runs": num_runs
            }
        }
        
        return results
    
    def print_performance_results(self, results: Dict):
        """Print performance comparison results"""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*60)
        
        vinn = results["vinn"]
        br = results["behavior_retrieval"]
        comp = results["comparison"]
        
        print(f"VINN:")
        print(f"  Average prediction time: {vinn['avg_time']:.4f} ± {vinn['std_time']:.4f} s")
        print(f"  Total time: {vinn['total_time']:.4f} s")
        
        print(f"\nBehavior Retrieval:")
        print(f"  Average prediction time: {br['avg_time']:.4f} ± {br['std_time']:.4f} s")
        print(f"  Total time: {br['total_time']:.4f} s")
        
        print(f"\nComparison:")
        print(f"  Speedup factor (BR/VINN): {comp['speedup_factor']:.2f}x")
        print(f"  Number of observations: {comp['num_observations']}")
        print(f"  Number of runs: {comp['num_runs']}")
        
        if comp['speedup_factor'] > 1:
            print(f"  -> VINN is {comp['speedup_factor']:.2f}x faster than Behavior Retrieval")
        else:
            print(f"  -> Behavior Retrieval is {1/comp['speedup_factor']:.2f}x faster than VINN")
    
    def run_full_comparison(self, k_neighbors: int = 5, num_test_images: int = 3):
        """
        Run full fair comparison suite
        """
        print("\n" + "="*60)
        print("FAIR COMPARISON: VINN vs BEHAVIOR RETRIEVAL")
        print("="*60)
        print("Objective: Same 2048-D BYOL embeddings, same action space")
        print("="*60)
        
        # Setup policies
        print("\n1. Setting up policies...")
        self.setup_vinn(k_neighbors=k_neighbors)
        self.setup_behavior_retrieval(train=True)
        
        # Generate test data
        print(f"\n2. Generating {num_test_images} test images...")
        test_images = []
        test_observations = []
        
        for i in range(num_test_images):
            # Generate random test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_images.append(test_image)
            test_observations.append({"image": test_image})
        
        # Verify identical embeddings
        print("\n3. Verifying identical BYOL embeddings...")
        embeddings_identical = self.verify_identical_embeddings(test_images)
        
        # Verify action space consistency  
        print("\n4. Verifying action space consistency...")
        actions_consistent = self.verify_action_space_consistency(test_observations)
        
        # Run performance comparison
        print("\n5. Running performance comparison...")
        performance_results = self.run_performance_comparison(test_observations, num_runs=3)
        self.print_performance_results(performance_results)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✓ Identical BYOL embeddings: {'PASS' if embeddings_identical else 'FAIL'}")
        print(f"✓ Consistent action space: {'PASS' if actions_consistent else 'FAIL'}")
        print(f"✓ Performance comparison: COMPLETED")
        
        if embeddings_identical and actions_consistent:
            print("\n🎉 FAIR COMPARISON SETUP VERIFIED!")
            print("Both methods use identical:")
            print("  - 2048-D BYOL embeddings from frozen ResNet-50")
            print("  - 7-DOF action space: Δ-pose (x,y,z,RPY) + gripper")
            print("  - RT-cache data processing pipeline")
        else:
            print("\n❌ FAIR COMPARISON SETUP FAILED!")
            print("Please fix the identified issues.")
        
        return {
            "embeddings_identical": embeddings_identical,
            "actions_consistent": actions_consistent,
            "performance_results": performance_results
        }


def main():
    """
    Run fair comparison between VINN and Behavior Retrieval
    """
    # Initialize comparison suite
    comparison = FairComparisonSuite(
        qdrant_host="localhost",
        qdrant_port=6333,
        mongo_host="mongodb://localhost:27017/",
        embedding_server="http://localhost:8000/predict",
        collection_name="image_collection"
    )
    
    # Run full comparison
    results = comparison.run_full_comparison(
        k_neighbors=5,
        num_test_images=3
    )
    
    print("\n[Main] Fair comparison completed.")
    return results


if __name__ == "__main__":
    main()