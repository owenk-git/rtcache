#!/usr/bin/env python3
"""
PROPER BehaviorRetrieval Training - Matching Original Paper Specifications
Based on Appendix A of the BehaviorRetrieval paper
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
import math

################################################################################
#                    PROPER BehaviorRetrieval Architecture
################################################################################

class SpatialSoftmax(nn.Module):
    """Spatial Softmax layer as in original paper"""
    def __init__(self, height, width, channel):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.width),
            np.linspace(-1., 1., self.height)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # feature: [B, C, H, W]
        batch_size = feature.size(0)
        
        # Flatten spatial dimensions: [B, C, H*W]
        feature = feature.view(batch_size, self.channel, -1)
        
        # Apply softmax over spatial dimension
        attention = F.softmax(feature, dim=-1)  # [B, C, H*W]
        
        # Compute expected positions
        expected_x = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)  # [B, C, 1]
        expected_y = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)  # [B, C, 1]
        
        # Concatenate x,y coordinates: [B, C*2]
        expected_xy = torch.cat([expected_x, expected_y], dim=-1).view(batch_size, -1)
        
        return expected_xy

class ProperVisualEncoder(nn.Module):
    """Original BehaviorRetrieval Visual Encoder: ResNet-18 + SpatialSoftmax → 64-dim"""
    def __init__(self):
        super(ProperVisualEncoder, self).__init__()
        
        # ResNet-18 backbone (remove final layers)
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & fc
        
        # Get feature map dimensions after backbone
        # For 84x84 input: ResNet-18 → [B, 512, ~3, ~3] feature maps
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 84, 84)
            dummy_output = self.backbone(dummy_input)
            _, C, H, W = dummy_output.shape
            print(f"ResNet-18 output shape: [{C}, {H}, {W}]")
        
        # Spatial Softmax: 512 channels → 512 * 2 = 1024 coordinates
        self.spatial_softmax = SpatialSoftmax(H, W, C)
        
        # Final MLP to get 64-dim embedding (as per paper)
        self.final_mlp = nn.Sequential(
            nn.Linear(C * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 84, 84] images
        Returns:
            [B, 64] visual embeddings
        """
        features = self.backbone(x)  # [B, 512, H, W]
        spatial_features = self.spatial_softmax(features)  # [B, 1024]
        embeddings = self.final_mlp(spatial_features)  # [B, 64]
        return embeddings

class ProperVAE(nn.Module):
    """Original BehaviorRetrieval VAE Architecture"""
    def __init__(self, visual_dim=64, action_dim=7, latent_dim=128, beta=0.0001):
        super(ProperVAE, self).__init__()
        
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.input_dim = visual_dim + action_dim  # 64 + 7 = 71
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: [300, 400] as per paper
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder: [300, 400] as per paper
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, self.input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def compute_embeddings(self, visual_features, actions):
        """Compute embeddings for retrieval (latent + actions)"""
        combined_input = torch.cat([visual_features, actions], dim=1)
        mu, logvar = self.encode(combined_input)
        # Return mu (latent) + actions concatenated
        embeddings = torch.cat([mu, actions], dim=1)  # [B, 128 + 7]
        return embeddings
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss with beta weighting"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

class MemoryAugmentedGMMPolicy(nn.Module):
    """Memory-Augmented GMM Policy as per original paper"""
    def __init__(self, visual_dim=64, action_dim=7, lstm_hidden=1000, lstm_horizon=10, n_modes=5):
        super(MemoryAugmentedGMMPolicy, self).__init__()
        
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_horizon = lstm_horizon
        self.n_modes = n_modes
        
        # Input dimension: visual + proprioception (assuming 7-DOF for simplicity)
        self.input_dim = visual_dim + action_dim  # 64 + 7 = 71
        
        # Two-layer LSTM with horizon 10
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True
        )
        
        # GMM output parameters: means, logvars, weights
        # Each mode needs: mean (action_dim) + logvar (action_dim) + weight (1)
        gmm_output_dim = n_modes * (action_dim * 2 + 1)
        
        self.gmm_head = nn.Sequential(
            nn.Linear(lstm_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, gmm_output_dim)
        )
        
    def forward(self, visual_seq, proprioception_seq):
        """
        Args:
            visual_seq: [B, T, visual_dim] visual features sequence
            proprioception_seq: [B, T, action_dim] proprioception sequence
        Returns:
            GMM parameters for the last timestep
        """
        # Concatenate visual and proprioception
        input_seq = torch.cat([visual_seq, proprioception_seq], dim=-1)  # [B, T, 71]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(input_seq)  # [B, T, lstm_hidden]
        
        # Use final timestep for action prediction
        final_hidden = lstm_out[:, -1, :]  # [B, lstm_hidden]
        
        # Generate GMM parameters
        gmm_params = self.gmm_head(final_hidden)  # [B, n_modes * (action_dim * 2 + 1)]
        
        # Parse GMM parameters
        batch_size = gmm_params.size(0)
        means = gmm_params[:, :self.n_modes * self.action_dim].view(batch_size, self.n_modes, self.action_dim)
        logvars = gmm_params[:, self.n_modes * self.action_dim:2 * self.n_modes * self.action_dim].view(batch_size, self.n_modes, self.action_dim)
        logits = gmm_params[:, 2 * self.n_modes * self.action_dim:]  # [B, n_modes]
        
        weights = F.softmax(logits, dim=-1)
        
        return means, logvars, weights
    
    def sample_action(self, means, logvars, weights, temperature=0.1):
        """Sample action from GMM with scaled variance"""
        batch_size = means.size(0)
        
        # Sample mode from categorical distribution
        mode_dist = torch.distributions.Categorical(weights)
        selected_modes = mode_dist.sample()  # [B]
        
        # Extract parameters for selected modes
        batch_indices = torch.arange(batch_size, device=means.device)
        selected_means = means[batch_indices, selected_modes]  # [B, action_dim]
        selected_logvars = logvars[batch_indices, selected_modes]  # [B, action_dim]
        
        # Scale down variance and sample
        scaled_vars = torch.exp(selected_logvars) * temperature
        action_dist = torch.distributions.Normal(selected_means, torch.sqrt(scaled_vars))
        actions = action_dist.sample()
        
        return actions

################################################################################
#                               Dataset Loading
################################################################################

DATASETS = [
    "berkeley_cable_routing", "roboturk", "nyu_door_opening_surprising_effectiveness", 
    "viola", "berkeley_autolab_ur5", "toto", "columbia_cairlab_pusht_real", 
    "austin_sirius_dataset_converted_externally_to_rlds", 
    "austin_sailor_dataset_converted_externally_to_rlds", 
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds", 
]

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'

class ProperOpenXDataset(Dataset):
    """Dataset with proper 84x84 images and sequential data for LSTM"""
    def __init__(self, datasets, max_samples_per_dataset=1000, sequence_length=10, device='cuda'):
        self.device = device
        self.sequence_length = sequence_length
        self.sequences = []
        
        print(f"Loading datasets with 84x84 images and sequence length {sequence_length}...")
        
        for dataset_name in datasets:
            print(f"Processing {dataset_name}...")
            try:
                builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
                ds = builder.as_dataset(split='train', shuffle_files=False)
                
                # Find image key
                if 'steps' in builder.info.features:
                    obs_keys = list(builder.info.features['steps']['observation'].keys())
                else:
                    obs_keys = list(builder.info.features.keys())
                
                possible_image_keys = ['image', 'rgb_static', 'front_rgb', 'agentview_rgb', 'rgb', 'hand_image', 'image_1']
                image_key = next((k for k in possible_image_keys if k in obs_keys), None)
                
                if not image_key:
                    print(f"No image key found for {dataset_name}, skipping")
                    continue
                
                count = 0
                current_sequence = []
                
                for episode in ds:
                    if 'steps' in episode:
                        steps = list(episode['steps'].as_numpy_iterator())
                    else:
                        steps = [episode]
                    
                    for step in steps:
                        if count >= max_samples_per_dataset:
                            break
                            
                        obs = step['observation'] if 'observation' in step else step
                        act = step['action'] if 'action' in step else np.zeros(7)
                        
                        # Process image to 84x84 (proper size)
                        image_data = obs[image_key]
                        if isinstance(image_data, tf.Tensor):
                            image_data = image_data.numpy()
                        
                        if image_data.dtype != np.uint8:
                            image_data = (image_data * 255).astype(np.uint8)
                        
                        image_pil = Image.fromarray(image_data).convert('RGB')
                        image_84 = image_pil.resize((84, 84), Image.LANCZOS)  # Proper 84x84 size
                        image_tensor = torch.FloatTensor(np.array(image_84)).permute(2, 0, 1) / 255.0
                        
                        # Process action
                        if isinstance(act, dict):
                            # Extract 7-DOF action
                            world_vector = act.get('world_vector', np.zeros(3))
                            rotation_delta = act.get('rotation_delta', np.zeros(3))
                            gripper = act.get('gripper_closedness_action', np.zeros(1))
                            action_vector = np.concatenate([world_vector, rotation_delta, gripper])
                        else:
                            action_vector = np.array(act)
                        
                        if len(action_vector) < 7:
                            action_vector = np.pad(action_vector, (0, 7 - len(action_vector)))
                        action_vector = action_vector[:7]
                        
                        current_sequence.append((image_tensor, torch.FloatTensor(action_vector)))
                        
                        # Create sequence when we have enough data
                        if len(current_sequence) >= self.sequence_length:
                            self.sequences.append(current_sequence[-self.sequence_length:])
                            current_sequence = current_sequence[1:]  # Sliding window
                        
                        count += 1
                        
                    if count >= max_samples_per_dataset:
                        break
                        
                print(f"Loaded {len(self.sequences)} sequences from {dataset_name}")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        images = torch.stack([item[0] for item in sequence])  # [T, 3, 84, 84]
        actions = torch.stack([item[1] for item in sequence])  # [T, 7]
        return images, actions

################################################################################
#                               Training
################################################################################

def train_proper_behavior_retrieval():
    """Train BehaviorRetrieval with proper architecture"""
    print("🚀 Training PROPER BehaviorRetrieval")
    print("✅ 84x84 images, ResNet-18+SpatialSoftmax→64D, Memory-augmented GMM")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    visual_encoder = ProperVisualEncoder().to(device)
    vae = ProperVAE(visual_dim=64, action_dim=7, latent_dim=128, beta=0.0001).to(device)
    policy = MemoryAugmentedGMMPolicy(visual_dim=64, action_dim=7).to(device)
    
    # Load dataset
    dataset = ProperOpenXDataset(
        datasets=DATASETS[:5],  # Start with fewer datasets
        max_samples_per_dataset=1000,
        sequence_length=10,
        device=device
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Batch size 16 as per paper
    
    # Optimizers
    vae_optimizer = torch.optim.Adam(list(visual_encoder.parameters()) + list(vae.parameters()), lr=1e-4)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Starting training...")
    
    # Training loop (simplified)
    visual_encoder.train()
    vae.train()
    policy.train()
    
    for epoch in range(10):  # Limited epochs for testing
        epoch_vae_loss = 0
        epoch_policy_loss = 0
        
        for batch_idx, (image_sequences, action_sequences) in enumerate(dataloader):
            image_sequences = image_sequences.to(device)  # [B, T, 3, 84, 84]
            action_sequences = action_sequences.to(device)  # [B, T, 7]
            
            batch_size, seq_len = image_sequences.shape[:2]
            
            # Step 1: Train VAE
            vae_optimizer.zero_grad()
            
            # Flatten sequences for VAE training
            images_flat = image_sequences.view(batch_size * seq_len, 3, 84, 84)
            actions_flat = action_sequences.view(batch_size * seq_len, 7)
            
            # Extract visual features
            visual_features = visual_encoder(images_flat)  # [B*T, 64]
            
            # VAE forward pass
            combined_input = torch.cat([visual_features, actions_flat], dim=1)
            recon_x, mu, logvar = vae(combined_input)
            
            vae_loss = vae.loss_function(recon_x, combined_input, mu, logvar)
            vae_loss.backward()
            vae_optimizer.step()
            
            # Step 2: Train Policy
            policy_optimizer.zero_grad()
            
            with torch.no_grad():
                visual_features_seq = visual_features.view(batch_size, seq_len, 64)
            
            # Policy forward pass (predict next action)
            means, logvars, weights = policy(visual_features_seq[:, :-1], action_sequences[:, :-1])
            target_actions = action_sequences[:, -1]  # Predict last action
            
            # Log-likelihood loss for GMM
            policy_loss = gmm_log_likelihood_loss(means, logvars, weights, target_actions)
            policy_loss.backward()
            policy_optimizer.step()
            
            epoch_vae_loss += vae_loss.item()
            epoch_policy_loss += policy_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: VAE Loss={vae_loss.item():.4f}, Policy Loss={policy_loss.item():.4f}")
        
        print(f"Epoch {epoch}: Avg VAE Loss={epoch_vae_loss/len(dataloader):.4f}, Avg Policy Loss={epoch_policy_loss/len(dataloader):.4f}")
    
    # Save models
    os.makedirs('./proper_br_models', exist_ok=True)
    torch.save(visual_encoder.state_dict(), './proper_br_models/visual_encoder.pth')
    torch.save(vae.state_dict(), './proper_br_models/vae.pth')
    torch.save(policy.state_dict(), './proper_br_models/policy.pth')
    
    print("✅ PROPER BehaviorRetrieval training completed!")

def gmm_log_likelihood_loss(means, logvars, weights, targets):
    """Compute log-likelihood loss for GMM"""
    batch_size, n_modes, action_dim = means.shape
    targets = targets.unsqueeze(1).expand(-1, n_modes, -1)  # [B, n_modes, action_dim]
    
    # Compute log probabilities for each mode
    log_probs = -0.5 * (logvars + (targets - means).pow(2) / torch.exp(logvars))
    log_probs = log_probs.sum(dim=-1)  # Sum over action dimensions
    
    # Weighted sum over modes
    weighted_log_probs = torch.log(weights) + log_probs
    log_likelihood = torch.logsumexp(weighted_log_probs, dim=-1)
    
    return -log_likelihood.mean()  # Negative log-likelihood

if __name__ == '__main__':
    train_proper_behavior_retrieval()