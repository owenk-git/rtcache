#!/usr/bin/env python3
"""
PROPER BehaviorRetrieval Implementation - Matching Original Algorithm
Based on the three-phase structure described in the original paper
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

################################################################################
#                    PHASE 1: VAE Similarity Metric Learning
################################################################################

class ProperBehaviorRetrievalVAE(nn.Module):
    """
    Original BehaviorRetrieval VAE for learning (s,a) similarity metric
    Matches equation (1) in the paper with β-VAE loss
    """
    def __init__(self, state_dim=64, action_dim=7, latent_dim=128, beta=0.0001):
        super(ProperBehaviorRetrievalVAE, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Separate encoders for state and action (as per paper)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined MLP for joint (s,a) embedding
        combined_dim = 128 + 64  # state + action features
        self.joint_encoder = nn.Sequential(
            nn.Linear(combined_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU()
        )
        
        # VAE latent layers
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder back to original state-action space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, state_dim + action_dim)
        )
        
    def encode_state_action(self, states, actions):
        """Encode state and action separately, then combine"""
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        combined = torch.cat([state_features, action_features], dim=1)
        joint_features = self.joint_encoder(combined)
        
        mu = self.fc_mu(joint_features)
        logvar = self.fc_logvar(joint_features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, states, actions):
        """
        Forward pass for VAE training
        Args:
            states: [B, state_dim] 
            actions: [B, action_dim]
        Returns:
            reconstructed [s,a], mu, logvar
        """
        mu, logvar = self.encode_state_action(states, actions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def compute_embeddings(self, states, actions):
        """
        Compute final embeddings for similarity computation
        Returns z = [z_sa, a] as described in the paper
        """
        mu, _ = self.encode_state_action(states, actions)
        # Concatenate latent with actions: z = [z_sa, a]
        embeddings = torch.cat([mu, actions], dim=1)
        return embeddings
    
    def similarity_function(self, states1, actions1, states2, actions2):
        """
        F(s1,a1,s2,a2) = -||z1 - z2||2 (Equation 2)
        """
        z1 = self.compute_embeddings(states1, actions1)
        z2 = self.compute_embeddings(states2, actions2)
        return -torch.norm(z1 - z2, dim=1, p=2)
    
    def elbo_loss(self, recon, target, mu, logvar):
        """ELBO loss with β weighting (Equation 1)"""
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

################################################################################
#                    PHASE 2: Data Retrieval (Filtering)
################################################################################

class BehaviorRetrieval:
    """
    Main BehaviorRetrieval class implementing the three-phase algorithm
    """
    def __init__(self, state_dim=64, action_dim=7, latent_dim=128, device='cuda'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize VAE for similarity metric
        self.vae = ProperBehaviorRetrievalVAE(state_dim, action_dim, latent_dim).to(device)
        
        # Storage for precomputed embeddings
        self.prior_embeddings = None
        self.prior_states = None
        self.prior_actions = None
        
        print("✅ BehaviorRetrieval initialized")
    
    def phase1_train_vae(self, prior_dataset, epochs=500, batch_size=50):
        """
        PHASE 1: Train VAE similarity metric on Dprior
        As described in Algorithm 1, lines 5-7
        """
        print(f"\n🔧 PHASE 1: Training VAE similarity metric on {len(prior_dataset)} samples")
        
        dataloader = DataLoader(prior_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        
        self.vae.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (states, actions) in enumerate(dataloader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                optimizer.zero_grad()
                
                # VAE forward pass
                target = torch.cat([states, actions], dim=1)  # Reconstruct [s,a]
                recon, mu, logvar = self.vae(states, actions)
                
                # ELBO loss
                loss = self.vae.elbo_loss(recon, target, mu, logvar)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.4f}")
        
        print("✅ Phase 1 Complete: VAE similarity metric trained")
        
        # Precompute embeddings for all prior data
        self._precompute_prior_embeddings(prior_dataset)
    
    def _precompute_prior_embeddings(self, prior_dataset):
        """Precompute embeddings for entire Dprior dataset"""
        print("🔄 Precomputing embeddings for prior dataset...")
        
        dataloader = DataLoader(prior_dataset, batch_size=100, shuffle=False)
        
        all_embeddings = []
        all_states = []
        all_actions = []
        
        self.vae.eval()
        with torch.no_grad():
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                embeddings = self.vae.compute_embeddings(states, actions)
                
                all_embeddings.append(embeddings.cpu())
                all_states.append(states.cpu())
                all_actions.append(actions.cpu())
        
        self.prior_embeddings = torch.cat(all_embeddings, dim=0)
        self.prior_states = torch.cat(all_states, dim=0)
        self.prior_actions = torch.cat(all_actions, dim=0)
        
        print(f"✅ Precomputed {len(self.prior_embeddings)} embeddings")
    
    def phase2_retrieve_data(self, task_dataset, delta=0.7):
        """
        PHASE 2: Retrieve relevant data from Dprior using Dt
        Implements Equation 3 from the paper
        """
        print(f"\n🔍 PHASE 2: Retrieving relevant data (δ={delta})")
        
        # Get task embeddings
        task_states = []
        task_actions = []
        for states, actions in DataLoader(task_dataset, batch_size=len(task_dataset)):
            task_states = states.to(self.device)
            task_actions = actions.to(self.device)
            break
        
        with torch.no_grad():
            task_embeddings = self.vae.compute_embeddings(task_states, task_actions)
        
        # Compute similarity matrix between all prior and task data
        prior_embeddings_gpu = self.prior_embeddings.to(self.device)
        
        print("Computing similarity matrix...")
        similarities = []
        
        # For each prior data point, find max similarity to any task data point
        batch_size = 1000  # Process in batches to save memory
        for i in range(0, len(prior_embeddings_gpu), batch_size):
            end_idx = min(i + batch_size, len(prior_embeddings_gpu))
            batch_prior = prior_embeddings_gpu[i:end_idx]
            
            # Compute distances to all task embeddings
            # F(s*,a*,s,a) = -||z1 - z2||2
            batch_similarities = []
            for task_emb in task_embeddings:
                dist = -torch.norm(batch_prior - task_emb.unsqueeze(0), dim=1, p=2)
                batch_similarities.append(dist)
            
            # Take max similarity for each prior sample
            batch_max_sim = torch.stack(batch_similarities, dim=1).max(dim=1)[0]
            similarities.append(batch_max_sim.cpu())
        
        all_similarities = torch.cat(similarities, dim=0)
        
        # Apply Equation 3: minmax normalization + threshold
        F_plus = all_similarities.max().item()
        F_minus = all_similarities.min().item()
        
        print(f"Similarity range: [{F_minus:.4f}, {F_plus:.4f}]")
        
        # Normalize and apply threshold
        normalized_similarities = (all_similarities - F_minus) / (F_plus - F_minus)
        retrieved_mask = normalized_similarities > delta
        
        # Extract retrieved data
        retrieved_states = self.prior_states[retrieved_mask]
        retrieved_actions = self.prior_actions[retrieved_mask]
        
        n_retrieved = retrieved_mask.sum().item()
        retrieval_rate = n_retrieved / len(self.prior_states)
        
        print(f"✅ Retrieved {n_retrieved} samples ({retrieval_rate:.2%} of prior data)")
        
        return retrieved_states, retrieved_actions
    
    def phase3_train_policy(self, task_dataset, retrieved_states, retrieved_actions, epochs=300):
        """
        PHASE 3: Train policy on union of Dt + Dret
        Standard behavior cloning as described in Equation 4
        """
        print(f"\n🎯 PHASE 3: Training policy on task + retrieved data")
        
        # Combine task and retrieved data
        task_states = []
        task_actions = []
        for states, actions in DataLoader(task_dataset, batch_size=len(task_dataset)):
            task_states = states
            task_actions = actions
            break
        
        # Union of datasets
        all_states = torch.cat([task_states, retrieved_states], dim=0)
        all_actions = torch.cat([task_actions, retrieved_actions], dim=0)
        
        print(f"Training data: {len(task_states)} task + {len(retrieved_states)} retrieved = {len(all_states)} total")
        
        # Simple policy network for demonstration
        policy = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        ).to(self.device)
        
        # Train policy with behavior cloning
        combined_dataset = torch.utils.data.TensorDataset(all_states, all_actions)
        dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        
        policy.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                optimizer.zero_grad()
                pred_actions = policy(states)
                loss = F.mse_loss(pred_actions, actions)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Policy Loss = {epoch_loss/len(dataloader):.6f}")
        
        print("✅ Phase 3 Complete: Policy trained on retrieved data")
        
        self.policy = policy
        return policy

################################################################################
#                               Dataset & Training
################################################################################

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

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'

class SimplifiedOpenXDataset(Dataset):
    """Simplified dataset for testing the algorithm"""
    def __init__(self, datasets, max_samples_per_dataset=1000, device='cuda', visual_encoder=None):
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
        
        print(f"Loading simplified dataset...")
        
        for dataset_name in datasets:  # Use ALL datasets for full training
            print(f"Processing {dataset_name}...")
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
                    continue
                
                count = 0
                for episode in ds:
                    if count >= max_samples_per_dataset:
                        break
                        
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
                
                print(f"Loaded {count} samples from {dataset_name}")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    """Run the complete BehaviorRetrieval algorithm"""
    print("🚀 PROPER BehaviorRetrieval - Three Phase Algorithm")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize BehaviorRetrieval
    br = BehaviorRetrieval(state_dim=64, action_dim=7, device=device)
    
    # Load datasets
    print("Loading datasets...")
    prior_dataset = SimplifiedOpenXDataset(DATASETS, max_samples_per_dataset=10000, device=device)
    task_dataset = SimplifiedOpenXDataset(DATASETS[-3:], max_samples_per_dataset=1000, device=device)  # 27-episode equivalent task dataset
    
    # Run three phases
    print("\n" + "="*60)
    print("STARTING THREE-PHASE BEHAVIOR RETRIEVAL")
    print("="*60)
    
    # Phase 1: Train VAE similarity metric
    br.phase1_train_vae(prior_dataset, epochs=500)  # Full VAE training
    
    # Phase 2: Retrieve relevant data
    retrieved_states, retrieved_actions = br.phase2_retrieve_data(task_dataset, delta=0.7)
    
    # Phase 3: Train policy
    policy = br.phase3_train_policy(task_dataset, retrieved_states, retrieved_actions, epochs=300)
    
    print("\n✅ PROPER BehaviorRetrieval training completed!")
    print("✅ Three-phase algorithm executed successfully")
    
    # Save models
    os.makedirs('./proper_br_models', exist_ok=True)
    torch.save(br.vae.state_dict(), './proper_br_models/vae_proper.pth')
    torch.save(policy.state_dict(), './proper_br_models/policy_proper.pth')
    
    print("✅ Models saved to ./proper_br_models/")

if __name__ == '__main__':
    main()