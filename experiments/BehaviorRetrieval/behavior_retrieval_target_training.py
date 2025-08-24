#!/usr/bin/env python3
"""
BehaviorRetrieval Training with Your 27-Episode Target Dataset
Modified to use your specific data structure with get_action_vector mapping
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
from pathlib import Path

# Import BehaviorRetrieval classes
from behavior_retrieval_proper import (
    ProperBehaviorRetrievalVAE, 
    BehaviorRetrieval,
    SimplifiedOpenXDataset,
    dataset2path
)

################################################################################
#                    Target Episode Dataset Loader
################################################################################

def get_action_vector(i: int, epi: str):
    """Your original Franka canned plan action vector generation"""
    def f(a,b,c,d,e): return \
        a if 1<=i<=5 else b if 6<=i<=8 else c if 9<=i<=12 else d if 13<=i<=17 else e
    _L1 = f([0, 0.035,0], [0,0,-0.055], [0,-0.02,0],  [0,0,-0.055], [0,0,0])
    _R1 = f([0,-0.035,0], [0,0,-0.055], [0, 0.02,0],  [0,0,-0.055], [0,0,0])
    _F1 = f([0.01,0,0],  [0,0,-0.055], [0,0.01,0],  [0,0,-0.055], [0,0,0])

    _L2 = f([0, 0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _R2 = f([0,-0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _F2 = f([0.02,0,0],  [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    
    _L3 = f([0, 0.035,0], [0,0,-0.055], [0, 0.01,0],  [0,0,-0.055], [0,0,0])
    _R3 = f([0,-0.035,0], [0,0,-0.055], [0, -0.01,0],  [0,0,-0.055], [0,0,0])
    _F3 = f([0.01,0,0],  [0,0,-0.055], [-0.01,0,0],  [0,0,-0.055], [0,0,0])

    families  = [[_L1,_L2,_L3], [_R1,_R2,_R3], [_F1,_F2,_F3]]

    try:
        eid = int(epi)
    except ValueError:
        return [0,0,0]
    if not 1<=eid<=28: return [0,0,0]
    fam  = (eid-1) % 3
    var  = ((eid-1)//3) % 3
    return families[fam][var]

def load_target_episodes(root_dir):
    """Load your 27-episode target dataset"""
    print(f"🎯 Loading 27-episode target dataset from {root_dir}")
    
    episodes_data = []
    successful_episodes = 0
    total_steps = 0
    
    for episode_id in range(1, 28):  # Episodes 1-27
        episode_path = Path(root_dir) / str(episode_id)
        if not episode_path.exists():
            continue
            
        episode_steps = 0
        
        for step in range(1, 18):  # Steps 1-17
            img_file = episode_path / f"{step:02d}.jpg"
            if not img_file.exists():
                img_file = episode_path / f"{step}.jpg"
            
            if not img_file.exists():
                continue
                
            try:
                image = Image.open(img_file).convert("RGB")
                action_3d = get_action_vector(step, str(episode_id))
                action_7d = np.array(action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
                episodes_data.append((image, action_7d))
                episode_steps += 1
                total_steps += 1
                
            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
                continue
        
        if episode_steps > 0:
            successful_episodes += 1
            print(f"✅ Episode {episode_id}: {episode_steps} steps loaded")
    
    print(f"\n📊 Dataset Summary: {successful_episodes}/27 episodes, {total_steps} total steps")
    return episodes_data

# Complete Open-X datasets for prior data (Dprior)
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

class TargetEpisodeDatasetForBR(Dataset):
    """
    Target episode dataset for BehaviorRetrieval (Dt)
    Extracts 64-D visual features using CNN encoder
    """
    
    def __init__(self, root_dir, device='cuda'):
        self.device = device
        
        # Load raw target episodes
        raw_data = load_target_episodes(root_dir)
        
        if len(raw_data) == 0:
            raise ValueError(f"No target episodes found in {root_dir}")
        
        # CNN encoder for 64-D visual features (matching BR paper)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64)
        ).to(device)
        
        # Precompute visual features
        print(f"🔄 Extracting 64-D visual features for {len(raw_data)} target samples...")
        
        self.visual_features = []
        self.actions = []
        
        self.visual_encoder.eval()
        with torch.no_grad():
            for i, (image, action) in enumerate(raw_data):
                # Convert PIL image to tensor (84x84 as per BR paper)
                image_resized = image.resize((84, 84), Image.LANCZOS)
                image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                # Extract 64-D visual features
                features = self.visual_encoder(image_tensor).cpu().squeeze(0)
                
                self.visual_features.append(features)
                self.actions.append(torch.FloatTensor(action))
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(raw_data)} samples...")
        
        print(f"✅ Target dataset prepared: {len(self.visual_features)} samples with 64-D features")
    
    def __len__(self):
        return len(self.visual_features)
    
    def __getitem__(self, idx):
        return self.visual_features[idx], self.actions[idx]

################################################################################
#                    BehaviorRetrieval Training with Target Episodes
################################################################################

def train_behavior_retrieval_with_targets(args):
    """Train BehaviorRetrieval using your 27-episode target dataset as Dt"""
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize BehaviorRetrieval
    br = BehaviorRetrieval(state_dim=64, action_dim=7, device=device)
    
    print("=" * 80)
    print("🎯 BEHAVIOR RETRIEVAL WITH 27-EPISODE TARGET DATASET")
    print("=" * 80)
    print(f"📁 Target directory: {args.target_dir}")
    print(f"📊 Prior datasets: {len(DATASETS)} Open-X datasets")
    print(f"🔧 VAE epochs: {args.vae_epochs}")
    print(f"🎯 Policy epochs: {args.policy_epochs}")
    print(f"🔍 Retrieval threshold δ: {args.delta}")
    print("=" * 80)
    
    # Load datasets
    print("\n📂 LOADING DATASETS...")
    print("-" * 50)
    
    # Phase 1: Load prior dataset (Dprior) from Open-X
    print(f"Loading prior dataset (Dprior) from {len(DATASETS)} Open-X datasets...")
    prior_dataset = SimplifiedOpenXDataset(
        DATASETS, 
        max_samples_per_dataset=args.prior_samples, 
        device=device
    )
    
    # Load your 27-episode target dataset (Dt)
    print(f"\nLoading your 27-episode target dataset (Dt)...")
    task_dataset = TargetEpisodeDatasetForBR(args.target_dir, device=device)
    
    if len(prior_dataset) == 0:
        print("❌ Failed to load prior data! Check Open-X dataset access.")
        return
    
    if len(task_dataset) == 0:
        print("❌ Failed to load target episodes! Check target directory.")
        return
    
    print(f"\n✅ Data loading complete:")
    print(f"  📊 Prior dataset (Dprior): {len(prior_dataset)} samples")
    print(f"  🎯 Target dataset (Dt): {len(task_dataset)} samples")
    
    # Run three phases
    print("\n" + "=" * 80)
    print("🔄 STARTING THREE-PHASE BEHAVIOR RETRIEVAL")
    print("=" * 80)
    
    total_start_time = time.time()
    
    # Phase 1: Train VAE similarity metric on Dprior
    print(f"\n🔧 PHASE 1: VAE Similarity Training on Open-X Prior Data")
    print(f"Training VAE on {len(prior_dataset)} prior samples for {args.vae_epochs} epochs...")
    phase1_start = time.time()
    
    br.phase1_train_vae(prior_dataset, epochs=args.vae_epochs, batch_size=args.batch_size)
    
    phase1_time = time.time() - phase1_start
    print(f"✅ Phase 1 completed in {phase1_time/60:.1f} minutes")
    
    # Phase 2: Retrieve relevant data using your 27-episode target dataset
    print(f"\n🔍 PHASE 2: Data Retrieval using Your 27-Episode Target Dataset")
    print(f"Using your target episodes to retrieve similar data from prior dataset...")
    print(f"Retrieval threshold δ = {args.delta}")
    phase2_start = time.time()
    
    retrieved_states, retrieved_actions = br.phase2_retrieve_data(task_dataset, delta=args.delta)
    
    phase2_time = time.time() - phase2_start
    retrieval_rate = len(retrieved_states) / len(prior_dataset)
    print(f"✅ Phase 2 completed in {phase2_time:.1f} seconds")
    print(f"✅ Retrieved {len(retrieved_states)} samples ({retrieval_rate:.2%} of prior data)")
    
    # Phase 3: Train policy on union of your target episodes + retrieved data
    print(f"\n🎯 PHASE 3: Policy Training on Target + Retrieved Data")
    print(f"Training BC policy on union of:")
    print(f"  🎯 Your target episodes: {len(task_dataset)} samples")
    print(f"  🔍 Retrieved data: {len(retrieved_states)} samples")
    print(f"  📈 Total training data: {len(task_dataset) + len(retrieved_states)} samples")
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
    print("🎉 BEHAVIOR RETRIEVAL WITH TARGET EPISODES COMPLETED!")
    print("=" * 80)
    print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
    print(f"📊 Final training composition:")
    print(f"  📂 Prior data (Dprior): {len(prior_dataset)} Open-X samples")
    print(f"  🎯 Target data (Dt): {len(task_dataset)} your episode samples")
    print(f"  🔍 Retrieved data: {len(retrieved_states)} filtered samples")
    print(f"  📈 BC policy trained on: {len(task_dataset) + len(retrieved_states)} samples")
    
    # Save models
    os.makedirs(args.save_dir, exist_ok=True)
    
    vae_path = f"{args.save_dir}/vae_target_training.pth"
    policy_path = f"{args.save_dir}/policy_target_training.pth"
    
    torch.save(br.vae.state_dict(), vae_path)
    torch.save(policy.state_dict(), policy_path)
    
    # Save training metadata
    metadata = {
        'target_dir': args.target_dir,
        'prior_datasets': DATASETS,
        'prior_samples_per_dataset': args.prior_samples,
        'target_episodes_loaded': len(task_dataset),
        'vae_epochs': args.vae_epochs,
        'policy_epochs': args.policy_epochs,
        'delta': args.delta,
        'retrieval_rate': retrieval_rate,
        'total_prior_samples': len(prior_dataset),
        'retrieved_samples': len(retrieved_states),
        'training_time_hours': total_time / 3600,
        'final_policy_training_samples': len(task_dataset) + len(retrieved_states)
    }
    
    with open(f"{args.save_dir}/target_training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Models saved:")
    print(f"  🧠 VAE model: {vae_path}")
    print(f"  🎯 Policy model: {policy_path}")
    print(f"  📋 Metadata: {args.save_dir}/target_training_metadata.json")
    
    return br, policy

def main():
    parser = argparse.ArgumentParser(description='BehaviorRetrieval Training with 27-Episode Target Dataset')
    
    # Dataset parameters
    parser.add_argument('--target_dir', type=str, 
                        default='./data/rt-cache/raw/',
                        help='Path to your 27-episode target dataset')
    parser.add_argument('--prior_samples', type=int, default=5000,
                        help='Max samples per Open-X dataset for prior data (default: 5000)')
    
    # Training parameters
    parser.add_argument('--vae_epochs', type=int, default=300,
                        help='VAE training epochs (default: 300)')
    parser.add_argument('--policy_epochs', type=int, default=200,
                        help='Policy training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--delta', type=float, default=0.7,
                        help='Retrieval threshold (default: 0.7)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='./br_target_models',
                        help='Directory to save models (default: ./br_target_models)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick_test:
        print("🧪 Running in QUICK TEST mode...")
        args.prior_samples = 500
        args.vae_epochs = 50
        args.policy_epochs = 50
        args.save_dir = './br_target_test'
    
    # Validate target directory
    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory not found: {args.target_dir}")
        print("Please check the path to your 27-episode dataset")
        return
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run training
    br, policy = train_behavior_retrieval_with_targets(args)
    
    print("\n🎯 Training completed successfully!")
    print(f"📁 Use models for evaluation:")
    print(f"  VAE: {args.save_dir}/vae_target_training.pth")
    print(f"  Policy: {args.save_dir}/policy_target_training.pth")

if __name__ == '__main__':
    main()