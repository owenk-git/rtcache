#!/usr/bin/env python3
"""
BehaviorRetrieval FULL TRAINING - Production Scale
Complete three-phase implementation with all Open-X datasets
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import pickle
import json

# Import the BehaviorRetrieval classes from the proper implementation
from behavior_retrieval_proper import (
    ProperBehaviorRetrievalVAE, 
    BehaviorRetrieval,
    SimplifiedOpenXDataset,
    dataset2path
)

# Complete Open-X datasets (all 19 for full training)
DATASETS = [
    "berkeley_cable_routing", "roboturk", "nyu_door_opening_surprising_effectiveness", 
    "viola", "berkeley_autolab_ur5", "toto", "columbia_cairlab_pusht_real", 
    "austin_sirius_dataset_converted_externally_to_rlds", 
    "austin_sailor_dataset_converted_externally_to_rlds", 
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds", 
    
    "tokyo_u_lsmo_converted_externally_to_rlds", 
    "dlr_sara_pour_converted_externally_to_rlds", "dlr_sara_grid_clamp_converted_externally_to_rlds", 
    "dlr_edan_shared_control_converted_externally_to_rlds", "asu_table_top_converted_externally_to_rlds", 
    "stanford_robocook_converted_externally_to_rlds", "utaustin_mutex",

    'fractal20220817_data',
    'kuka', 
    'bridge'
]

class FullOpenXDataset(Dataset):
    """
    Full-scale OpenX dataset loader for production training
    Uses ALL datasets without the [:3] limitation
    """
    def __init__(self, datasets, max_samples_per_dataset=10000, device='cuda', visual_encoder=None):
        self.device = device
        self.data = []
        
        # Use dummy visual encoder if none provided
        if visual_encoder is None:
            visual_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 64)
            ).to(device)
        
        print(f"Loading FULL OpenX dataset from {len(datasets)} datasets...")
        print(f"Max samples per dataset: {max_samples_per_dataset}")
        
        successful_datasets = 0
        
        for dataset_idx, dataset_name in enumerate(datasets):
            print(f"\n[{dataset_idx+1}/{len(datasets)}] Processing {dataset_name}...")
            start_time = time.time()
            
            try:
                builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
                ds = builder.as_dataset(split='train', shuffle_files=False)
                
                # Find image key
                if 'steps' in builder.info.features:
                    obs_keys = list(builder.info.features['steps']['observation'].keys())
                else:
                    obs_keys = list(builder.info.features.keys())
                
                possible_image_keys = ['image', 'rgb_static', 'front_rgb', 'agentview_rgb']
                image_key = next((k for k in possible_image_keys if k in obs_keys), None)
                
                if not image_key:
                    print(f"  ❌ No valid image key found, skipping {dataset_name}")
                    continue
                
                count = 0
                episode_count = 0
                
                for episode in ds:
                    if count >= max_samples_per_dataset:
                        break
                        
                    episode_count += 1
                    
                    if 'steps' in episode:
                        steps = list(episode['steps'].as_numpy_iterator())
                    else:
                        steps = [episode]
                    
                    for step in steps:
                        if count >= max_samples_per_dataset:
                            break
                            
                        obs = step['observation'] if 'observation' in step else step
                        act = step['action'] if 'action' in step else np.zeros(7)
                        
                        # Process image and extract visual features
                        image_data = obs[image_key]
                        if isinstance(image_data, tf.Tensor):
                            image_data = image_data.numpy()
                        
                        if image_data.dtype != np.uint8:
                            image_data = (image_data * 255).astype(np.uint8)
                        
                        # Ensure RGB format
                        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                            image_pil = Image.fromarray(image_data).convert('RGB')
                            image_resized = image_pil.resize((84, 84), Image.LANCZOS)
                            image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
                            
                            # Extract visual features
                            with torch.no_grad():
                                visual_features = visual_encoder(image_tensor.unsqueeze(0).to(device)).cpu().squeeze(0)
                            
                            # Process action
                            if isinstance(act, dict):
                                world_vector = act.get('world_vector', np.zeros(3))
                                rotation_delta = act.get('rotation_delta', np.zeros(3))
                                gripper = act.get('gripper_closedness_action', np.zeros(1))
                                action_vector = np.concatenate([world_vector, rotation_delta, gripper])
                            else:
                                action_vector = np.array(act)
                            
                            if len(action_vector) < 7:
                                action_vector = np.pad(action_vector, (0, 7 - len(action_vector)))
                            action_vector = action_vector[:7]
                            
                            self.data.append((visual_features, torch.FloatTensor(action_vector)))
                            count += 1
                
                duration = time.time() - start_time
                print(f"  ✅ Loaded {count} samples from {episode_count} episodes in {duration:.2f}s")
                successful_datasets += 1
                
            except Exception as e:
                print(f"  ❌ Error loading {dataset_name}: {e}")
                continue
        
        print(f"\n🎯 DATASET LOADING COMPLETE:")
        print(f"  📊 Successfully loaded {successful_datasets}/{len(datasets)} datasets")
        print(f"  📈 Total samples: {len(self.data)}")
        print(f"  💾 Memory usage: ~{len(self.data) * 64 * 4 / 1024 / 1024:.1f} MB")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def full_behavior_retrieval_training(args):
    """Run complete BehaviorRetrieval training with production parameters"""
    
    print("🚀 BEHAVIOR RETRIEVAL - FULL PRODUCTION TRAINING")
    print("=" * 80)
    print(f"📊 Datasets: {len(DATASETS)} Open-X datasets")
    print(f"📈 Prior data: {args.prior_samples} samples per dataset")
    print(f"🎯 Task data: {args.task_samples} samples")
    print(f"🔧 VAE epochs: {args.vae_epochs}")
    print(f"🎯 Policy epochs: {args.policy_epochs}")
    print(f"📱 Device: {args.device}")
    print("=" * 80)
    
    device = torch.device(args.device)
    
    # Initialize BehaviorRetrieval
    br = BehaviorRetrieval(state_dim=64, action_dim=7, device=device)
    
    # Load datasets
    print("\n📂 LOADING DATASETS...")
    print("-" * 50)
    
    print(f"Loading prior dataset (Dprior) from {len(DATASETS)} datasets...")
    prior_dataset = FullOpenXDataset(
        DATASETS, 
        max_samples_per_dataset=args.prior_samples, 
        device=device
    )
    
    print(f"\nLoading task dataset (Dt) from last {args.task_datasets} datasets...")
    task_dataset = FullOpenXDataset(
        DATASETS[-args.task_datasets:], 
        max_samples_per_dataset=args.task_samples, 
        device=device
    )
    
    if len(prior_dataset) == 0 or len(task_dataset) == 0:
        print("❌ Failed to load sufficient data! Check dataset access.")
        return
    
    print(f"\n✅ Data loading complete:")
    print(f"  📊 Prior dataset: {len(prior_dataset)} samples")
    print(f"  🎯 Task dataset: {len(task_dataset)} samples")
    
    # Run three phases
    print("\n" + "=" * 80)
    print("🔄 STARTING THREE-PHASE BEHAVIOR RETRIEVAL")
    print("=" * 80)
    
    total_start_time = time.time()
    
    # Phase 1: Train VAE similarity metric
    print(f"\n🔧 PHASE 1: VAE Similarity Metric Training")
    print(f"Training on {len(prior_dataset)} prior samples for {args.vae_epochs} epochs...")
    phase1_start = time.time()
    
    br.phase1_train_vae(prior_dataset, epochs=args.vae_epochs, batch_size=args.batch_size)
    
    phase1_time = time.time() - phase1_start
    print(f"✅ Phase 1 completed in {phase1_time/60:.1f} minutes")
    
    # Phase 2: Retrieve relevant data
    print(f"\n🔍 PHASE 2: Data Retrieval")
    print(f"Retrieving relevant data with δ={args.delta}...")
    phase2_start = time.time()
    
    retrieved_states, retrieved_actions = br.phase2_retrieve_data(task_dataset, delta=args.delta)
    
    phase2_time = time.time() - phase2_start
    print(f"✅ Phase 2 completed in {phase2_time:.1f} seconds")
    
    # Phase 3: Train policy
    print(f"\n🎯 PHASE 3: Policy Training")
    print(f"Training policy for {args.policy_epochs} epochs...")
    phase3_start = time.time()
    
    policy = br.phase3_train_policy(
        task_dataset, 
        retrieved_states, 
        retrieved_actions, 
        epochs=args.policy_epochs
    )
    
    phase3_time = time.time() - phase3_start
    print(f"✅ Phase 3 completed in {phase3_time/60:.1f} minutes")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 80)
    print("🎉 BEHAVIOR RETRIEVAL TRAINING COMPLETED!")
    print("=" * 80)
    print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
    print(f"📊 Final dataset sizes:")
    print(f"  📂 Prior data: {len(prior_dataset)} samples")
    print(f"  🎯 Task data: {len(task_dataset)} samples")
    print(f"  🔍 Retrieved data: {len(retrieved_states)} samples")
    print(f"  📈 Total training data: {len(task_dataset) + len(retrieved_states)} samples")
    
    # Save models
    os.makedirs(args.save_dir, exist_ok=True)
    
    vae_path = f"{args.save_dir}/vae_full_training.pth"
    policy_path = f"{args.save_dir}/policy_full_training.pth"
    
    torch.save(br.vae.state_dict(), vae_path)
    torch.save(policy.state_dict(), policy_path)
    
    # Save training metadata
    metadata = {
        'datasets': DATASETS,
        'prior_samples_per_dataset': args.prior_samples,
        'task_samples': args.task_samples,
        'vae_epochs': args.vae_epochs,
        'policy_epochs': args.policy_epochs,
        'delta': args.delta,
        'total_prior_samples': len(prior_dataset),
        'total_task_samples': len(task_dataset),
        'retrieved_samples': len(retrieved_states),
        'training_time_hours': total_time / 3600
    }
    
    with open(f"{args.save_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Models saved:")
    print(f"  🧠 VAE model: {vae_path}")
    print(f"  🎯 Policy model: {policy_path}")
    print(f"  📋 Metadata: {args.save_dir}/training_metadata.json")
    
    return br, policy

def main():
    parser = argparse.ArgumentParser(description='BehaviorRetrieval Full Training')
    
    # Dataset parameters
    parser.add_argument('--prior_samples', type=int, default=10000,
                        help='Max samples per dataset for prior data (default: 10000)')
    parser.add_argument('--task_samples', type=int, default=1000,
                        help='Max samples for task dataset (default: 1000)')
    parser.add_argument('--task_datasets', type=int, default=3,
                        help='Number of datasets to use for task data (default: 3)')
    
    # Training parameters
    parser.add_argument('--vae_epochs', type=int, default=500,
                        help='VAE training epochs (default: 500)')
    parser.add_argument('--policy_epochs', type=int, default=300,
                        help='Policy training epochs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size (default: 50)')
    parser.add_argument('--delta', type=float, default=0.7,
                        help='Retrieval threshold (default: 0.7)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='./br_full_models',
                        help='Directory to save models (default: ./br_full_models)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick_test:
        print("🧪 Running in QUICK TEST mode...")
        args.prior_samples = 1000
        args.task_samples = 100
        args.vae_epochs = 50
        args.policy_epochs = 50
        args.save_dir = './br_test_models'
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run training
    br, policy = full_behavior_retrieval_training(args)
    
    print("\n🎯 Training completed successfully!")
    print(f"Use these models for evaluation:")
    print(f"  VAE: {args.save_dir}/vae_full_training.pth")
    print(f"  Policy: {args.save_dir}/policy_full_training.pth")

if __name__ == '__main__':
    main()