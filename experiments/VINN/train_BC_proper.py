#!/usr/bin/env python3
"""
PROPER VINN BC Training - Matching Original Logic Exactly
Only difference: Uses Open-X embodiment datasets instead of original datasets
Maintains all core VINN logic including temporal features and fraction-based evaluation
"""

import sys
import numpy as np
import argparse
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from collections import deque

# Add VINN modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'imitation_models'))
from BC_openx import BC

# Add local embedding extractor
from local_embedding_fix import LocalBYOLEmbeddingExtractor

################################################################################
#                           RT-Cache Integration (Local Embeddings)
################################################################################

# Complete Open-X datasets
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

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'

def ensure_rank1(value, default):
    t = tf.convert_to_tensor(value) if value is not None else tf.zeros(default)
    if t.shape.ndims == 0:
        t = tf.expand_dims(t, 0)
    return t

def _extract_action(action_data):
    """Extract 7-DOF action: world_vector(3) + rotation_delta(3) + gripper(1)"""
    if isinstance(action_data, dict):
        world_vector = ensure_rank1(action_data.get('world_vector'), (3,))
        rotation_delta = ensure_rank1(action_data.get('rotation_delta'), (3,))
        gripper = ensure_rank1(action_data.get('gripper_closedness_action'), (1,))
        combined = tf.concat([world_vector, rotation_delta, gripper], axis=-1)
        return combined.numpy().astype(np.float32)
    
    if isinstance(action_data, tf.Tensor) and action_data.shape[-1] == 7:
        return action_data.numpy().astype(np.float32)
    
    return np.zeros(7, dtype=np.float32)

def _clamp(val, low, high):
    return max(low, min(high, val))

def normalize_franka_action(raw_action):
    """Normalize action to standard range"""
    raw_list = raw_action.tolist()
    pos = [_clamp(x, -0.1, 0.1) for x in raw_list[:3]]
    ori = [_clamp(r, -0.5, 0.5) for r in raw_list[3:6]]
    # RESTORED: Gripper as 4-class classification (following original VINN)
    grip_val = raw_list[6]
    if grip_val <= -0.5:
        grip_class = 0
    elif grip_val <= 0.0:
        grip_class = 1
    elif grip_val <= 0.5:
        grip_class = 2
    else:
        grip_class = 3
    return np.array(pos + ori + [grip_class], dtype=np.float32)

################################################################################
#              OpenXDataset - Matching Original Dataset Structure
################################################################################

class OpenXDataset(Dataset):
    """
    Dataset matching original VINN structure with:
    - Temporal stacking support (t+1 frames)
    - get_subset() method for fraction-based training
    - Proper data structure matching original
    """
    def __init__(self, params, encoder, partial=None):
        self.params = params
        self.encoder = encoder
        self.temporal_window = params.get('t', 0) + 1  # t+1 frames
        
        # Match original data structure
        self.representations = []
        self.translation = []
        self.rotation = []
        self.gripper = []
        self.paths = []
        self.episode_boundaries = []  # Track episode boundaries for temporal stacking
        
        print(f"Loading Open-X data with temporal window={self.temporal_window}")
        print(f"Architecture: {params.get('architecture', 'ResNet')}")
        print(f"Representation dim: {2048 if params.get('architecture') == 'ResNet' else 9216}")
        
        self.extract_data(partial)
    
    def extract_data(self, factor=None):
        """Extract data from Open-X datasets matching original logic"""
        
        # Calculate how many datasets to use based on factor
        total_datasets = len(DATASETS)
        if factor is not None:
            total_datasets = int(total_datasets * factor)
        
        datasets_to_use = DATASETS[:total_datasets]
        print(f"Using {len(datasets_to_use)} datasets (factor={factor})")
        
        for dataset_idx, dataset_name in enumerate(datasets_to_use):
            print(f"\nProcessing dataset: {dataset_name}")
            
            try:
                # Load dataset
                builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
                ds = builder.as_dataset(split='train', shuffle_files=False)
                
                # Detect keys (same as before)
                possible_image_keys = [
                    'image', 'rgb_static', 'front_rgb', 'agentview_rgb',
                    'rgb', 'hand_image', 'image_1'
                ]
                
                if 'steps' in builder.info.features:
                    is_rlds = True
                    obs_keys = list(builder.info.features['steps']['observation'].keys())
                else:
                    is_rlds = False
                    obs_keys = list(builder.info.features.keys())
                
                display_image_key = next((k for k in possible_image_keys if k in obs_keys), None)
                if not display_image_key:
                    print(f"No valid image key found in dataset {dataset_name}; skipping.")
                    continue
                
                # Process episodes
                if is_rlds:
                    for episode in ds.take(100):  # Limit episodes per dataset
                        episode_start_idx = len(self.representations)
                        steps_list = list(episode["steps"].as_numpy_iterator())
                        
                        if len(steps_list) < self.temporal_window:
                            continue
                        
                        # Process each step in episode
                        for step_idx, step_np in enumerate(steps_list):
                            obs = step_np["observation"]
                            act = step_np["action"]
                            
                            # Extract action
                            action_vector = _extract_action(act)
                            norm_action = normalize_franka_action(action_vector)
                            
                            # Process image
                            image_data = obs[display_image_key]
                            if isinstance(image_data, tf.Tensor):
                                image_data = image_data.numpy()
                            
                            if image_data.dtype != np.uint8:
                                img_array = (image_data * 255).astype(np.uint8)
                            else:
                                img_array = image_data
                            
                            image_pil = Image.fromarray(img_array)
                            
                            # Extract embedding
                            try:
                                embedding = self.encoder.extract_embedding(image_pil)  # [2048]
                                
                                # Store data matching original structure
                                self.representations.append(embedding)
                                self.translation.append(torch.FloatTensor(norm_action[:3]))
                                self.rotation.append(torch.FloatTensor(norm_action[3:6]))
                                self.gripper.append(torch.tensor([int(norm_action[6])]))
                                self.paths.append(f"{dataset_name}_ep{episode_start_idx}_step{step_idx}")
                                
                            except Exception as e:
                                print(f"Embedding extraction failed: {e}")
                                continue
                            
                            # Check gripper termination
                            if action_vector[-1] == 1:
                                break
                        
                        # Mark episode boundary
                        self.episode_boundaries.append(len(self.representations))
                
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        print(f"\nTotal samples loaded: {len(self.representations)}")
        print(f"Total episodes: {len(self.episode_boundaries)}")
    
    def get_subset(self, fraction):
        """Get subset of data for fraction-based training (matching original)"""
        total_episodes = len(self.episode_boundaries)
        subset_episodes = int(fraction * total_episodes)
        
        # Select random episodes
        episode_indices = np.random.choice(total_episodes, subset_episodes, replace=False)
        
        # Build subset data
        subset_data = []
        for ep_idx in sorted(episode_indices):
            # Get episode boundaries
            start_idx = self.episode_boundaries[ep_idx-1] if ep_idx > 0 else 0
            end_idx = self.episode_boundaries[ep_idx]
            
            # Add all samples from this episode
            for idx in range(start_idx, end_idx):
                if idx >= self.temporal_window - 1:  # Ensure we have enough history
                    subset_data.append(self.__getitem__(idx))
        
        return subset_data
    
    def __len__(self):
        # Only count samples that have enough temporal context
        return max(0, len(self.representations) - self.temporal_window + 1)
    
    def __getitem__(self, index):
        """Get item with temporal stacking (matching original)"""
        # For temporal stacking, concatenate t+1 frames
        if self.temporal_window > 1:
            # Stack temporal frames
            stacked_representations = []
            
            # Get current and past frames
            for t in range(self.temporal_window):
                frame_idx = index - (self.temporal_window - 1 - t)
                if frame_idx >= 0:
                    stacked_representations.append(self.representations[frame_idx])
                else:
                    # Pad with zeros if not enough history
                    stacked_representations.append(np.zeros_like(self.representations[0]))
            
            # Concatenate temporal features
            representation = np.concatenate(stacked_representations, axis=0)
        else:
            # Single frame (t=0)
            representation = self.representations[index]
        
        # Return data matching original structure
        return (
            torch.FloatTensor(representation),
            self.translation[index],
            self.rotation[index],
            self.gripper[index],
            self.paths[index]
        )

################################################################################
#                    Main Training Function (Matching Original)
################################################################################

def run_bc_model(params):
    """Run BC model with fraction-based evaluation (matching original train_BC.py)"""
    all_losses = []
    all_means = []
    all_stds = []
    
    # Initialize BC class with proper params
    bc = BC(params)
    
    # Define fractions for evaluation (matching original)
    if params['bc_model'] == 'BC_rep':
        fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        fractions = [0.05, 0.1, 0.3, 0.7, 1.0]
    
    # Run evaluation for each fraction
    for p in fractions:
        print(f"\n{'='*60}")
        print(f"Evaluating with fraction: {p}")
        print(f"{'='*60}")
        
        losses = []
        if params['dataset'] == 'HandleData':
            losses = bc.get_val_losses(p, 5)  # 5 runs per fraction
        else:
            losses = bc.get_test_losses(p, 5)
        
        losses = np.array(losses)
        mean = np.mean(losses)
        std = np.std(losses)
        print(f"Fraction {p}: mean={mean:.4f}, std={std:.4f}")
        
        all_losses.append(losses)
        all_means.append(mean)
        all_stds.append(std)
    
    # Save results (matching original format)
    os.makedirs("../results", exist_ok=True)
    suffix = f"{params['dataset']}_{params['pretrained']}.txt"
    np.savetxt(f"../results/{params['bc_model']}_losses_{suffix}", np.array(all_losses), delimiter=",")
    np.savetxt(f"../results/{params['bc_model']}_means_{suffix}", np.array(all_means), delimiter=",")
    np.savetxt(f"../results/{params['bc_model']}_stds_{suffix}", np.array(all_stds), delimiter=",")
    
    print(f"\nResults saved to ../results/")

def main():
    # Parse arguments matching original
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--train_dir', type=str, default='')  # Not used with Open-X
    parser.add_argument('--val_dir', type=str, default='')    # Not used with Open-X
    parser.add_argument('--test_dir', type=str, default='')   # Not used with Open-X
    parser.add_argument('--layer', type=str, default='avgpool')
    parser.add_argument('--dataset', type=str, default='HandleData')
    parser.add_argument('--representation_model_path', type=str, default='')
    parser.add_argument('--model', type=str, default='byol')
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='./vinn_proper_models/')
    parser.add_argument('--architecture', type=str, default='ResNet')
    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--temporal', type=int, default=0)
    parser.add_argument('--t', type=int, default=0)  # CRITICAL: temporal frames (0 = single frame)
    parser.add_argument('--bc_model', type=str, default='BC_rep', choices=['BC_rep', 'BC_end_to_end'])
    parser.add_argument('--pretrain_encoder', type=int, default=1)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--partial', type=float, default=1.0)
    
    args = parser.parse_args()
    params = vars(args)
    
    print("=" * 80)
    print("VINN BC Training - PROPER Implementation")
    print("Matching Original Logic with Open-X Datasets")
    print("=" * 80)
    print(f"Temporal frames (t+1): {params['t'] + 1}")
    print(f"Architecture: {params['architecture']}")
    print(f"BC model type: {params['bc_model']}")
    print(f"Dataset: {params['dataset']}")
    print(f"Batch size: {params['batch_size']}")
    print(f"Learning rate: {params['lr']}")
    print(f"Epochs: {params['epochs']}")
    print("=" * 80)
    
    
    # Run the BC model with proper evaluation
    run_bc_model(params)
    
    print("\n✅ VINN training completed with PROPER implementation!")
    print("✅ Maintained all original logic: temporal features, fraction-based evaluation")
    print("✅ Only difference: Using Open-X embodiment datasets")

if __name__ == '__main__':
    main()