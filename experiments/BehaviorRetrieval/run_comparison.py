#!/usr/bin/env python3
"""
Run Fair Comparison between VINN and Behavior Retrieval
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fair_comparison import FairComparisonSuite
from rt_cache_config import RTCacheConfig


def main():
    print("=" * 60)
    print("FAIR COMPARISON: VINN vs BEHAVIOR RETRIEVAL")
    print("=" * 60)
    print("This will verify both methods use identical:")
    print("  • 2048-D BYOL embeddings from frozen ResNet-50")
    print("  • 7-DOF action space: Δ-pose (x,y,z,RPY) + gripper")
    print("  • RT-cache data processing pipeline")
    print("=" * 60)
    
    # Initialize comparison
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
    
    return results


if __name__ == "__main__":
    main()