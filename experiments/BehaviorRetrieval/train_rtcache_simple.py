#!/usr/bin/env python3
"""
Simplified Behavior Retrieval Training using RT-cache data processing pipeline
No robomimic dependency - uses PyTorch directly
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
import requests
from io import BytesIO
import base64
import time

################################################################################
#                           RT-Cache Integration
################################################################################

REMOTE_SERVER_URL = "http://localhost:8000/predict"

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
    grip = 1.0 if raw_list[6] > 0 else 0.0
    return np.array(pos + ori + [grip], dtype=np.float32)

def send_for_embedding(image_pil, text_prompt=None, url=REMOTE_SERVER_URL, option="image"):
    """Get OpenVLA/CLIP embeddings from remote server"""
    files = {}
    if image_pil is not None:
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        buf.seek(0)
        files["file"] = ("image.png", buf, "image/png")

    data = {
        "instruction": text_prompt if text_prompt else "",
        "option": option
    }
    resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json()

def decode_base64_torch_tensor(b64_string):
    binary_data = base64.b64decode(b64_string)
    buff = BytesIO(binary_data)
    try:
        tensor = torch.load(buff, map_location="cpu")
        # Convert BFloat16 to Float32 if needed
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor
    except Exception as e:
        print(f"Error loading tensor: {e}")
        # Return zeros if loading fails
        return torch.zeros(2176, dtype=torch.float32)

################################################################################
#                           Simplified Neural Networks
################################################################################

class VAE(nn.Module):
    """Shallow VAE for re-embedding state-action pairs"""
    def __init__(self, input_dim=2176, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class BehaviorCloning(nn.Module):
    """BC head for action prediction"""
    def __init__(self, latent_dim=128, action_dim=7):
        super(BehaviorCloning, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

################################################################################
#                           Dataset Class
################################################################################

class OpenXDataset(Dataset):
    """RT-cache integrated dataset for Behavior Retrieval"""
    def __init__(self, datasets=DATASETS, max_samples_per_dataset=1000):
        self.data = []
        
        print(f"Loading Open-X data from {len(datasets)} datasets using RT-cache pipeline...")
        
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
                
                # Process episodes/samples
                if is_rlds:
                    for episode in ds:
                        if sample_count >= max_samples_per_dataset:
                            break
                            
                        episode_idx += 1
                        steps_dataset = episode["steps"]
                        steps_list = list(steps_dataset.as_numpy_iterator())
                        
                        total_steps_in_episode = len(steps_list)
                        if total_steps_in_episode < 5:
                            continue
                        
                        step_idx_local = 0
                        
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
                            
                            # Process image
                            image_data = obs[display_image_key]
                            if isinstance(image_data, tf.Tensor):
                                image_data = image_data.numpy()
                            
                            if image_data.dtype != np.uint8:
                                img_array = (image_data * 255).astype(np.uint8)
                            else:
                                img_array = image_data
                            
                            image_pil = Image.fromarray(img_array)
                            
                            # Get embedding from server
                            try:
                                server_out = send_for_embedding(image_pil, None, option="image")
                                image_features_b64 = server_out.get("image_features", None)
                                if image_features_b64:
                                    image_tensor = decode_base64_torch_tensor(image_features_b64)
                                    embedding = image_tensor.squeeze(0).numpy()  # [2176]
                                else:
                                    continue
                            except Exception as e:
                                print(f"Remote call failed, skipping. Error: {e}")
                                continue
                            
                            self.data.append({
                                'embedding': embedding,
                                'action': norm_action,
                                'dataset': dataset_name
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
#                           Training Functions
################################################################################

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function"""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_behavior_retrieval(args):
    """Train Behavior Retrieval with VAE + BC"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading Open-X dataset via RT-cache pipeline...")
    dataset = OpenXDataset(
        datasets=DATASETS[:3],  # Use first 3 datasets for demo
        max_samples_per_dataset=args.max_samples
    )
    
    if len(dataset) == 0:
        print("❌ No data loaded! Check RT-cache server connection.")
        return
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize models
    vae = VAE(input_dim=2176, latent_dim=128).to(device)
    bc = BehaviorCloning(latent_dim=128, action_dim=7).to(device)
    
    # Optimizers
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    bc_optimizer = torch.optim.Adam(bc.parameters(), lr=args.lr)
    
    # Train VAE
    print(f"\n[VAE] Training VAE for {args.vae_epochs} epochs...")
    vae.train()
    
    for epoch in range(args.vae_epochs):
        total_loss = 0
        for embeddings, actions in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}"):
            embeddings = embeddings.to(device)
            
            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = vae(embeddings)
            loss = vae_loss(recon_batch, embeddings, mu, logvar)
            loss.backward()
            vae_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[VAE] Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Train BC
    print(f"\n[BC] Training BC head for {args.bc_epochs} epochs...")
    vae.eval()
    bc.train()
    criterion = nn.MSELoss()
    
    for epoch in range(args.bc_epochs):
        total_loss = 0
        for embeddings, actions in tqdm(train_loader, desc=f"BC Epoch {epoch+1}"):
            embeddings = embeddings.to(device)
            actions = actions.to(device)
            
            # Get VAE latent representation
            with torch.no_grad():
                mu, logvar = vae.encode(embeddings)
                z = vae.reparameterize(mu, logvar)
            
            bc_optimizer.zero_grad()
            predicted_actions = bc(z)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            bc_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[BC] Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    # Save models
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(vae.state_dict(), f"{args.save_dir}/vae.pth")
    torch.save(bc.state_dict(), f"{args.save_dir}/bc.pth")
    
    print(f"✅ Models saved to {args.save_dir}")
    
    # Quick evaluation
    print("\n[EVAL] Quick evaluation on validation set...")
    vae.eval()
    bc.eval()
    
    total_error = 0
    with torch.no_grad():
        for embeddings, actions in val_loader:
            embeddings = embeddings.to(device)
            actions = actions.to(device)
            
            mu, logvar = vae.encode(embeddings)
            z = vae.reparameterize(mu, logvar)
            predicted_actions = bc(z)
            
            error = F.mse_loss(predicted_actions, actions)
            total_error += error.item()
    
    avg_error = total_error / len(val_loader)
    print(f"Validation MSE: {avg_error:.4f}")
    
    return vae, bc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--vae_epochs', type=int, default=20)
    parser.add_argument('--bc_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='./br_simple_models')
    parser.add_argument('--max_samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Simplified Behavior Retrieval Training with RT-Cache Open-X Data")
    print("=" * 80)
    print(f"VAE epochs: {args.vae_epochs}")
    print(f"BC epochs: {args.bc_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)
    
    # Train models
    models = train_behavior_retrieval(args)
    
    print("✅ Behavior Retrieval training completed!")

if __name__ == '__main__':
    main()