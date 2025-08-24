#!/usr/bin/env python3
"""
VINN BC Training with LOCAL embeddings (no RT-cache server dependency)
Solves BFloat16 issues by using local ResNet-50 BYOL embeddings
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

# Add VINN modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'imitation_models'))
from BC import TranslationModel, RotationModel, GripperModel

# Add local embedding extractor (now in same directory)
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
#                     LOCAL Embedding Data Loading
################################################################################

class OpenXDataset(Dataset):
    """
    Dataset loader using LOCAL embeddings (no server dependency)
    Uses frozen ResNet-50 BYOL embeddings for fair comparison
    """
    def __init__(self, datasets=DATASETS, max_samples_per_dataset=1000, device='cuda'):
        self.data = []
        self.device = device
        
        print(f"Loading Open-X data from {len(datasets)} datasets using LOCAL embeddings...")
        print("✅ Using frozen ResNet-50 BYOL embeddings (following research objective)")
        
        # Initialize local embedding extractor
        self.embedding_extractor = LocalBYOLEmbeddingExtractor(device=device)
        
        exclusion_dataset_list = []
        
        for dataset_idx, dataset_name in enumerate(datasets):
            if dataset_name in exclusion_dataset_list:
                print(f"Skipping dataset: {dataset_name}")
                continue
            
            print(f"\n========== Processing Dataset: {dataset_name} ==========")
            start_time = time.time()
            
            try:
                # EXACT same TFDS loading as RT-cache
                builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
                ds = builder.as_dataset(split='train', shuffle_files=False)
                
                # EXACT same key detection logic
                possible_image_keys = [
                    'image', 'rgb_static', 'front_rgb', 'agentview_rgb',
                    'rgb', 'hand_image', 'image_1'
                ]
                possible_text_keys = ['natural_language_instruction', 'language_instruction']
                
                # EXACT same RLDS detection
                if 'steps' in builder.info.features:
                    is_rlds = True
                    obs_keys = list(builder.info.features['steps']['observation'].keys())
                else:
                    is_rlds = False
                    obs_keys = list(builder.info.features.keys())
                
                display_image_key = next((k for k in possible_image_keys if k in obs_keys), None)
                display_text_key = next((k for k in possible_text_keys if k in obs_keys), None)
                if not display_image_key:
                    print(f"No valid image key found in dataset {dataset_name}; skipping.")
                    continue
                
                point_idx = 0
                episode_idx = 0
                COUNT = 0
                sample_count = 0
                
                # ------------------- (A) RLDS CASE -------------------
                if is_rlds:
                    for episode in ds:
                        if sample_count >= max_samples_per_dataset:
                            break
                            
                        episode_idx += 1
                        steps_dataset = episode["steps"]
                        steps_list = list(steps_dataset.as_numpy_iterator())
                        
                        total_steps_in_episode = len(steps_list)
                        if total_steps_in_episode < 5:
                            print(f"Skipping dataset={dataset_name}, ep={episode_idx}, <5 steps.")
                            continue
                        
                        step_idx_local = 0
                        episode_text = None
                        
                        first_obs = steps_list[0]["observation"]
                        if display_text_key and (display_text_key in first_obs):
                            episode_text = first_obs[display_text_key].decode("utf-8")
                        
                        for step_np in steps_list:
                            if sample_count >= max_samples_per_dataset:
                                break
                                
                            step_idx_local += 1
                            point_idx += 1
                            COUNT += 1
                            
                            obs = step_np["observation"]
                            act = step_np["action"]
                            action_vector = _extract_action(act)
                            norm_action = normalize_franka_action(action_vector)
                            
                            doc_id = f"{dataset_name}_{episode_idx}_{step_idx_local}"
                            
                            # EXACT same image processing as RT-cache
                            image_data = obs[display_image_key]
                            
                            if isinstance(image_data, tf.Tensor):
                                image_data = image_data.numpy()
                            
                            if image_data.dtype != np.uint8:
                                img_array = (image_data * 255).astype(np.uint8)
                            else:
                                img_array = image_data
                            
                            image_pil = Image.fromarray(img_array)
                            
                            # LOCAL embedding extraction (replaces server call)
                            try:
                                embedding = self.embedding_extractor.extract_embedding(image_pil)  # [2176]
                            except Exception as e:
                                print(f"Local embedding failed for doc_id={doc_id}, skipping. Error: {e}")
                                continue
                            
                            self.data.append({
                                'embedding': embedding,
                                'action': norm_action,
                                'dataset': dataset_name,
                                'doc_id': doc_id
                            })
                            sample_count += 1
                            
                            # EXACT same gripper termination logic
                            if action_vector[-1] == 1:
                                break
                            
                            if COUNT > 100:
                                COUNT = 0
                
                # ------------------- (B) NON-RLDS CASE -------------------
                else:
                    ds_list = list(ds)
                    total_steps_in_episode = len(ds_list)
                    if total_steps_in_episode < 5:
                        print(f"Skipping {dataset_name}, <5 samples")
                        continue
                    
                    episode_idx = 1
                    step_idx_local = 0
                    COUNT = 0
                    episode_text = None
                    
                    if len(ds_list) > 0 and display_text_key in ds_list[0]:
                        sample0 = ds_list[0]
                        episode_text = sample0[display_text_key].numpy().decode("utf-8")
                    
                    for sample in tqdm(ds_list, desc=f"Steps in {dataset_name}"):
                        if sample_count >= max_samples_per_dataset:
                            break
                            
                        step_idx_local += 1
                        point_idx += 1
                        COUNT += 1
                        
                        action_data = sample.get('action', tf.zeros(7))
                        action_vector = _extract_action(action_data)
                        norm_action = normalize_franka_action(action_vector)
                        
                        doc_id = f"{dataset_name}_{episode_idx}_{step_idx_local}"
                        
                        # EXACT same image processing
                        image_data = sample[display_image_key]
                        if isinstance(image_data, tf.Tensor):
                            image_data = image_data.numpy()
                        
                        if image_data.dtype != np.uint8:
                            img_array = (image_data * 255).astype(np.uint8)
                        else:
                            img_array = image_data
                        
                        image_pil = Image.fromarray(img_array)
                        
                        # LOCAL embedding extraction
                        try:
                            embedding = self.embedding_extractor.extract_embedding(image_pil)
                        except Exception as e:
                            print(f"Local embedding failed for doc_id={doc_id}, skipping. Error: {e}")
                            continue
                        
                        self.data.append({
                            'embedding': embedding,
                            'action': norm_action,
                            'dataset': dataset_name,
                            'doc_id': doc_id
                        })
                        sample_count += 1
                        
                        if action_vector[-1] == 1:
                            break
                        
                        if COUNT > 100:
                            COUNT = 0
                
                duration = time.time() - start_time
                print(f"[INFO] Done with dataset={dataset_name} in {duration:.2f} seconds.")
                print(f"Loaded {sample_count} samples from {dataset_name}")
                
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        print(f"[INFO] Total samples loaded: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.FloatTensor(item['embedding'])
        action = torch.FloatTensor(item['action'])
        return embedding, action

################################################################################
#                           Training
################################################################################

def train_vinn_bc(args):
    """Train VINN BC models using LOCAL embeddings"""
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading Open-X dataset with LOCAL BYOL embeddings...")
    dataset = OpenXDataset(
        datasets=DATASETS,  # Use all 19 datasets for proper training
        max_samples_per_dataset=args.max_samples,
        device=device
    )
    
    if len(dataset) == 0:
        print("❌ No data loaded! Check dataset access.")
        return
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize models (using 2048-D BYOL embeddings as per research objective)
    embedding_dim = 2048
    translation_model = TranslationModel(embedding_dim).to(device)
    rotation_model = RotationModel(embedding_dim).to(device)
    gripper_model = GripperModel(embedding_dim).to(device)
    
    # Optimizers
    trans_optimizer = torch.optim.Adam(translation_model.parameters(), lr=args.lr)
    rot_optimizer = torch.optim.Adam(rotation_model.parameters(), lr=args.lr)
    grip_optimizer = torch.optim.Adam(gripper_model.parameters(), lr=args.lr)
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()  # For gripper classification
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        translation_model.train()
        rotation_model.train()
        gripper_model.train()
        
        train_trans_loss = 0
        train_rot_loss = 0
        train_grip_loss = 0
        
        for embeddings, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            embeddings = embeddings.to(device)
            actions = actions.to(device)
            
            # Split actions
            trans_target = actions[:, :3]  # xyz
            rot_target = actions[:, 3:6]   # RPY
            grip_target = actions[:, 6].long()  # gripper as class indices (0-3)
            
            # Forward
            trans_pred = translation_model(embeddings)
            rot_pred = rotation_model(embeddings)
            grip_pred = gripper_model(embeddings)
            
            # Losses
            trans_loss = mse_criterion(trans_pred, trans_target)
            rot_loss = mse_criterion(rot_pred, rot_target)
            grip_loss = ce_criterion(grip_pred, grip_target)  # CrossEntropy for classification
            
            # Backward
            trans_optimizer.zero_grad()
            trans_loss.backward()
            trans_optimizer.step()
            
            rot_optimizer.zero_grad()
            rot_loss.backward()
            rot_optimizer.step()
            
            grip_optimizer.zero_grad()
            grip_loss.backward()
            grip_optimizer.step()
            
            train_trans_loss += trans_loss.item()
            train_rot_loss += rot_loss.item()
            train_grip_loss += grip_loss.item()
        
        # Validation
        translation_model.eval()
        rotation_model.eval()
        gripper_model.eval()
        
        val_trans_loss = 0
        val_rot_loss = 0
        val_grip_loss = 0
        
        with torch.no_grad():
            for embeddings, actions in val_loader:
                embeddings = embeddings.to(device)
                actions = actions.to(device)
                
                trans_target = actions[:, :3]
                rot_target = actions[:, 3:6]
                grip_target = actions[:, 6].long()  # gripper as class indices
                
                trans_pred = translation_model(embeddings)
                rot_pred = rotation_model(embeddings)
                grip_pred = gripper_model(embeddings)
                
                val_trans_loss += mse_criterion(trans_pred, trans_target).item()
                val_rot_loss += mse_criterion(rot_pred, rot_target).item()
                val_grip_loss += ce_criterion(grip_pred, grip_target).item()
        
        # Print losses
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Trans: {train_trans_loss/len(train_loader):.4f}, "
              f"Rot: {train_rot_loss/len(train_loader):.4f}, "
              f"Grip: {train_grip_loss/len(train_loader):.4f}")
        print(f"  Val   - Trans: {val_trans_loss/len(val_loader):.4f}, "
              f"Rot: {val_rot_loss/len(val_loader):.4f}, "
              f"Grip: {val_grip_loss/len(val_loader):.4f}")
    
    # Save models
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(translation_model.state_dict(), f"{args.save_dir}/translation_model.pth")
    torch.save(rotation_model.state_dict(), f"{args.save_dir}/rotation_model.pth")
    torch.save(gripper_model.state_dict(), f"{args.save_dir}/gripper_model.pth")
    
    print(f"Models saved to {args.save_dir}")
    
    return translation_model, rotation_model, gripper_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./vinn_local_models')
    parser.add_argument('--max_samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VINN BC Training with LOCAL BYOL Embeddings (No Server Dependency)")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)
    
    # Train models
    models = train_vinn_bc(args)
    
    print("✅ VINN BC training completed with LOCAL embeddings!")

if __name__ == '__main__':
    main()