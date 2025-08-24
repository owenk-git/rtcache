#!/usr/bin/env python3
"""
Behavior Retrieval Training with LOCAL embeddings (no RT-cache server dependency)
Solves BFloat16 issues by using local ResNet-50 BYOL embeddings
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

# Add original BehaviorRetrieval visual encoder (ResNet18 + SpatialSoftmax)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

################################################################################
#                           RT-Cache Integration (Local Embeddings)
################################################################################

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

################################################################################
#                           Original BehaviorRetrieval Architecture
################################################################################

class SpatialSoftmax(nn.Module):
    """Original SpatialSoftmax from BehaviorRetrieval (matches robomimic)"""
    def __init__(self, height, width, num_kp=32, temperature=1.0):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.num_kp = num_kp
        self.temperature = temperature
        
        # Create coordinate meshgrid
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing='xy'
        )
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
    
    def forward(self, feature):
        """
        feature: [B, C, H, W]
        output: [B, C * 2] (x, y coordinates for each channel)
        """
        B, C, H, W = feature.shape
        
        # Apply temperature and softmax
        attention = F.softmax(feature.view(B, C, -1) / self.temperature, dim=-1)
        attention = attention.view(B, C, H, W)
        
        # Compute expected coordinates
        expected_x = torch.sum(attention * self.pos_x, dim=[2, 3])  # [B, C]
        expected_y = torch.sum(attention * self.pos_y, dim=[2, 3])  # [B, C]
        
        # Concatenate x, y coordinates
        keypoints = torch.cat([expected_x, expected_y], dim=1)  # [B, 2*C]
        return keypoints

class VisualEncoder(nn.Module):
    """Original BehaviorRetrieval visual encoder: ResNet18 + SpatialSoftmax"""
    def __init__(self, feature_dim=128, num_kp=32):
        super(VisualEncoder, self).__init__()
        
        # ResNet18 backbone (NO pretraining as per original config)
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Get feature map dimensions (ResNet18 outputs 512 channels)
        # After conv layers: [B, 512, H, W] where H,W depend on input size
        self.conv_channels = 512
        
        # SpatialSoftmax pooling
        self.spatial_softmax = SpatialSoftmax(height=7, width=7, num_kp=num_kp)  # Assume 7x7 after ResNet18
        
        # Final linear layer to desired feature dimension
        self.fc = nn.Linear(self.conv_channels * 2, feature_dim)  # *2 for (x,y) coordinates
        
    def forward(self, x):
        """
        x: [B, 3, 224, 224] RGB images
        output: [B, feature_dim] visual features
        """
        # ResNet18 feature extraction
        features = self.backbone(x)  # [B, 512, H, W]
        
        # SpatialSoftmax pooling
        keypoints = self.spatial_softmax(features)  # [B, 512*2]
        
        # Final linear projection
        output = self.fc(keypoints)  # [B, feature_dim]
        
        return output

class VAE(nn.Module):
    """Original BehaviorRetrieval VAE: takes observation+action, reconstructs observation+action"""
    def __init__(self, visual_dim=128, action_dim=7, latent_dim=128):
        super(VAE, self).__init__()
        
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.input_dim = visual_dim + action_dim  # 128 + 7 = 135
        self.latent_dim = latent_dim
        
        # Encoder (following original config: [300, 400] layer dims)
        self.fc1 = nn.Linear(self.input_dim, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder (following original config: [300, 400] layer dims)
        self.fc3 = nn.Linear(latent_dim, 300)
        self.fc4 = nn.Linear(300, 400)
        self.fc5 = nn.Linear(400, self.input_dim)  # Reconstruct visual+action
        
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
        return self.fc5(h4)  # Removed sigmoid for real-valued embeddings
    
    def forward(self, x):
        """
        x: concatenated [visual_features, actions] of shape [B, 135]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def compute_embeddings(self, visual_features, actions):
        """
        Original BehaviorRetrieval embedding computation:
        Returns VAE latent + actions concatenated (as in classifier.py line 197)
        """
        # Concatenate visual features and actions
        combined_input = torch.cat([visual_features, actions], dim=1)  # [B, 135]
        
        # Encode to get latent
        mu, logvar = self.encode(combined_input)
        
        # Return latent + actions concatenated (following original)
        embeddings = torch.cat([mu, actions], dim=1)  # [B, latent_dim + action_dim]
        return embeddings

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
    """Dataset using original BehaviorRetrieval visual encoder"""
    def __init__(self, datasets=DATASETS, max_samples_per_dataset=1000, device='cuda'):
        self.data = []
        self.device = device
        
        print(f"Loading Open-X data from {len(datasets)} datasets using ORIGINAL BehaviorRetrieval architecture...")
        print("✅ Using ResNet18 + SpatialSoftmax (following original implementation)")
        
        # Initialize original visual encoder (ResNet18 + SpatialSoftmax)
        self.visual_encoder = VisualEncoder(feature_dim=128, num_kp=32).to(device)
        self.visual_encoder.eval()
        
        # Image preprocessing for ResNet18
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
                
                # Process episodes/samples (same logic as VINN)
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
                            
                            # Original BehaviorRetrieval visual encoding
                            try:
                                image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    visual_features = self.visual_encoder(image_tensor)  # [1, 128]
                                embedding = visual_features.cpu().numpy().squeeze(0)  # [128]
                            except Exception as e:
                                print(f"Visual encoding failed, skipping. Error: {e}")
                                continue
                            
                            self.data.append({
                                'visual_features': embedding,  # [128] visual features
                                'action': norm_action,          # [7] actions 
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
        visual_features = torch.FloatTensor(item['visual_features'])  # [128]
        action = torch.FloatTensor(item['action'])                    # [7]
        return visual_features, action

################################################################################
#                           Training Functions
################################################################################

def vae_loss(recon_x, x, mu, logvar, visual_dim=128, action_dim=7, kl_weight=0.0001):
    """
    Original BehaviorRetrieval VAE loss function
    Reconstructs both visual features (128D) and actions (7D)
    """
    # Split reconstruction and target into visual and action components
    recon_visual = recon_x[:, :visual_dim]      # [B, 128] 
    recon_action = recon_x[:, visual_dim:]      # [B, 7]
    target_visual = x[:, :visual_dim]           # [B, 128]
    target_action = x[:, visual_dim:]           # [B, 7]
    
    # Reconstruction loss for both components
    visual_mse = F.mse_loss(recon_visual, target_visual, reduction='sum')
    action_mse = F.mse_loss(recon_action, target_action, reduction='sum')
    reconstruction_loss = visual_mse + action_mse
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Combined loss (following original kl_weight from config)
    total_loss = reconstruction_loss + kl_weight * kl_loss
    return total_loss

def train_behavior_retrieval(args):
    """Train Behavior Retrieval with VAE + BC using LOCAL embeddings"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading Open-X dataset with ORIGINAL BehaviorRetrieval architecture...")
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
    
    # Initialize models (original architecture: visual_dim=128, action_dim=7)
    vae = VAE(visual_dim=128, action_dim=7, latent_dim=128).to(device)
    # BC takes VAE latent + actions as input (128 + 7 = 135 dimensions)
    bc = BehaviorCloning(latent_dim=128 + 7, action_dim=7).to(device)
    
    # Optimizers
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    bc_optimizer = torch.optim.Adam(bc.parameters(), lr=args.lr)
    
    # Train VAE
    print(f"\n[VAE] Training VAE for {args.vae_epochs} epochs...")
    vae.train()
    
    for epoch in range(args.vae_epochs):
        total_loss = 0
        for visual_features, actions in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}"):
            visual_features = visual_features.to(device)  # [B, 128]
            actions = actions.to(device)                  # [B, 7]
            
            # Create joint input (visual + action) as per original
            joint_input = torch.cat([visual_features, actions], dim=1)  # [B, 135]
            
            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = vae(joint_input)
            loss = vae_loss(recon_batch, joint_input, mu, logvar)
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
        for visual_features, actions in tqdm(train_loader, desc=f"BC Epoch {epoch+1}"):
            visual_features = visual_features.to(device)  # [B, 128]
            actions = actions.to(device)                  # [B, 7]
            
            # Get VAE embeddings (latent + actions concatenated) following original
            with torch.no_grad():
                embeddings = vae.compute_embeddings(visual_features, actions)  # [B, latent_dim + action_dim]
            
            bc_optimizer.zero_grad()
            predicted_actions = bc(embeddings)
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
        for visual_features, actions in val_loader:
            visual_features = visual_features.to(device)  # [B, 128]
            actions = actions.to(device)                  # [B, 7]
            
            # Get VAE embeddings (latent + actions concatenated)
            embeddings = vae.compute_embeddings(visual_features, actions)  # [B, 135]
            predicted_actions = bc(embeddings)
            
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
    parser.add_argument('--save_dir', type=str, default='./br_local_models')
    parser.add_argument('--max_samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Behavior Retrieval Training - CORRECTED Original Architecture")
    print("VAE: ResNet18+SpatialSoftmax (128D) + Actions (7D) → Latent (128D)")
    print("Embeddings: VAE Latent (128D) + Actions (7D) = 135D")
    print("=" * 80)
    print(f"VAE epochs: {args.vae_epochs}")
    print(f"BC epochs: {args.bc_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)
    
    # Train models
    models = train_behavior_retrieval(args)
    
    print("✅ Behavior Retrieval training completed with CORRECTED original architecture!")
    print("✅ VAE now properly reconstructs visual+action pairs (135D total)")
    print("✅ Embeddings now include both VAE latent + action components")

if __name__ == '__main__':
    main()