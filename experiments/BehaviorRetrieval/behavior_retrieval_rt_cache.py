"""
Behavior Retrieval Implementation
Integrated with RT-cache data processing pipeline

Key features:
- Uses 2048-D BYOL embeddings from RT-cache  
- Re-embeds state-action pairs with shallow VAE
- Retrieves ~25% of Open-X data
- Fine-tunes BC head
- Identical action space: Δ-pose (x,y,z,RPY) + gripper
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Dict, Optional, Any
import random
from collections import defaultdict

# Vector database and MongoDB
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pymongo import MongoClient

# Machine learning
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class RTCacheDataset(Dataset):
    """
    Dataset that loads 2048-D BYOL embeddings and actions from RT-cache
    """
    
    def __init__(
        self, 
        qdrant_client: QdrantClient,
        mongo_collection,
        collection_name: str,
        sample_fraction: float = 0.25,
        max_samples: int = None
    ):
        """
        Initialize dataset with RT-cache data
        
        Args:
            qdrant_client: Qdrant client for vector retrieval
            mongo_collection: MongoDB collection for metadata
            collection_name: Qdrant collection name  
            sample_fraction: Fraction of data to sample (~25% for BR)
            max_samples: Maximum number of samples to load
        """
        self.qdrant_client = qdrant_client
        self.mongo_collection = mongo_collection
        self.collection_name = collection_name
        self.sample_fraction = sample_fraction
        
        # Load data from RT-cache
        self.data_points = self._load_rt_cache_data(max_samples)
        print(f"[BehaviorRetrieval] Loaded {len(self.data_points)} data points")
    
    def _load_rt_cache_data(self, max_samples: int = None) -> List[Dict]:
        """
        Load data points from RT-cache (Qdrant + MongoDB)
        """
        print(f"[BehaviorRetrieval] Loading {self.sample_fraction*100:.1f}% of RT-cache data...")
        
        # First, get all point IDs from Qdrant
        all_points = []
        offset = 0
        batch_size = 100
        
        while True:
            try:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    offset=offset,
                    limit=batch_size,
                    with_vectors=True,
                    with_payload=True
                )
                
                if not points:
                    break
                
                all_points.extend(points)
                
                if next_offset is None or (max_samples and len(all_points) >= max_samples):
                    break
                    
                offset = next_offset
                
            except Exception as e:
                print(f"[BehaviorRetrieval] Error loading data: {e}")
                break
        
        # Sample fraction of the data
        if self.sample_fraction < 1.0:
            sample_size = int(len(all_points) * self.sample_fraction)
            all_points = random.sample(all_points, sample_size)
        
        if max_samples:
            all_points = all_points[:max_samples]
        
        # Process points into dataset format
        data_points = []
        for point in all_points:
            try:
                # Get embedding from Qdrant
                embedding = np.array(point.vector, dtype=np.float32)
                
                # Get action from MongoDB
                logical_id = point.payload.get("logical_id", None)
                if not logical_id:
                    continue
                
                doc = self.mongo_collection.find_one(
                    {"id": logical_id}, 
                    {"norm_action": 1, "raw_action": 1, "dataset_name": 1}
                )
                
                if not doc or "norm_action" not in doc:
                    continue
                
                action = np.array(doc["norm_action"], dtype=np.float32)
                
                # Ensure 7-DOF action
                if action.shape[0] != 7:
                    continue
                
                data_point = {
                    "embedding": embedding,
                    "action": action,
                    "logical_id": logical_id,
                    "dataset_name": doc.get("dataset_name", "unknown"),
                    "payload": point.payload
                }
                data_points.append(data_point)
                
            except Exception as e:
                print(f"[BehaviorRetrieval] Error processing point: {e}")
                continue
        
        return data_points
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        """
        Get data point for training
        Returns: (embedding, action) pair
        """
        data_point = self.data_points[idx]
        return {
            "embedding": torch.from_numpy(data_point["embedding"]).float(),
            "action": torch.from_numpy(data_point["action"]).float(),
            "metadata": data_point
        }


class ShallowVAE(nn.Module):
    """
    Shallow VAE for re-embedding state-action pairs
    Following the Behavior Retrieval approach
    """
    
    def __init__(
        self,
        input_dim: int = 2055,  # 2048 (BYOL) + 7 (action)
        latent_dim: int = 128,
        hidden_dims: List[int] = [512, 256]
    ):
        """
        Initialize shallow VAE
        
        Args:
            input_dim: Input dimension (2048 BYOL + 7 action = 2055)
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
        """
        super(ShallowVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decode latent to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def get_embedding(self, x):
        """Get latent embedding (for retrieval)"""
        mu, _ = self.encode(x)
        return mu


class BehaviorCloning(nn.Module):
    """
    Behavior Cloning head that takes VAE embeddings and predicts actions
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        action_dim: int = 7,
        hidden_dims: List[int] = [256, 256]
    ):
        """
        Initialize BC head
        
        Args:
            embedding_dim: Input embedding dimension from VAE
            action_dim: Output action dimension (7-DOF)
            hidden_dims: Hidden layer dimensions
        """
        super(BehaviorCloning, self).__init__()
        
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, embedding):
        """Predict action from embedding"""
        return self.network(embedding)


class BehaviorRetrievalPolicy:
    """
    Behavior Retrieval Policy integrated with RT-cache
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        mongo_host: str = "mongodb://localhost:27017/",
        embedding_server: str = "http://localhost:8000/predict",
        collection_name: str = "image_collection",
        sample_fraction: float = 0.25,
        vae_latent_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Behavior Retrieval policy
        
        Args:
            qdrant_host: Qdrant vector database host
            qdrant_port: Qdrant port
            mongo_host: MongoDB connection string
            embedding_server: RT-cache embedding server URL
            collection_name: Qdrant collection name
            sample_fraction: Fraction of Open-X to retrieve (~25%)
            vae_latent_dim: VAE latent dimension
            device: Training device
        """
        
        self.device = torch.device(device)
        self.embedding_server = embedding_server
        self.sample_fraction = sample_fraction
        self.vae_latent_dim = vae_latent_dim
        
        # Initialize connections
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60.0)
        self.mongo_client = MongoClient(mongo_host)
        self.db = self.mongo_client["OpenVLACollection"]  
        self.collection = self.db["OpenVLACollection"]
        self.collection_name = collection_name
        
        # Initialize models
        self.vae = ShallowVAE(
            input_dim=2055,  # 2048 BYOL + 7 action
            latent_dim=vae_latent_dim
        ).to(self.device)
        
        self.bc_head = BehaviorCloning(
            embedding_dim=vae_latent_dim,
            action_dim=7
        ).to(self.device)
        
        # Training state
        self.is_trained = False
        
        print(f"[BehaviorRetrieval] Initialized with sample_fraction={sample_fraction}")
        print(f"[BehaviorRetrieval] VAE latent dim: {vae_latent_dim}")
        print(f"[BehaviorRetrieval] Device: {device}")
    
    def _get_byol_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Get 2048-D BYOL embedding from RT-cache embedding server
        Identical to VINN implementation for fair comparison
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
            
        # Preprocess like RT-cache
        image_pil = self._center_crop_to_square(image_pil)
        image_pil = image_pil.resize((224, 224), Image.Resampling.LANCZOS)
        
        try:
            response = self._send_for_byol_embedding(image_pil)
            
            if "byol_features" in response:
                byol_b64 = response["byol_features"]
            elif "image_features" in response:
                byol_b64 = response["image_features"]
            else:
                raise ValueError("No BYOL features found")
                
            embedding_tensor = self._decode_base64_torch_tensor(byol_b64)
            embedding = embedding_tensor.squeeze(0).numpy().astype(np.float32)
            
            if embedding.shape[0] != 2048:
                raise ValueError(f"Expected 2048-D BYOL embedding, got {embedding.shape[0]}-D")
                
            return embedding
            
        except Exception as e:
            print(f"[BehaviorRetrieval] Error getting BYOL embedding: {e}")
            return np.zeros(2048, dtype=np.float32)
    
    def _center_crop_to_square(self, pil_image: Image.Image) -> Image.Image:
        """Center crop image to square"""
        w, h = pil_image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return pil_image.crop((left, top, left + side, top + side))
    
    def _send_for_byol_embedding(self, pil_image: Image.Image) -> Dict:
        """Send image to RT-cache embedding server"""
        buf = BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)
        
        files = {"file": ("image.jpg", buf, "image/jpeg")}
        data = {"instruction": "", "option": "byol"}
        
        response = requests.post(self.embedding_server, files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()
    
    def _decode_base64_torch_tensor(self, b64_string: str) -> torch.Tensor:
        """Decode base64-encoded PyTorch tensor"""
        binary_data = base64.b64decode(b64_string)
        buff = BytesIO(binary_data)
        tensor = torch.load(buff, map_location="cpu")
        return tensor
    
    def train_vae(
        self, 
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_samples: int = 100000
    ):
        """
        Train the shallow VAE on RT-cache data
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_samples: Maximum samples to use for training
        """
        print(f"[BehaviorRetrieval] Training VAE for {num_epochs} epochs...")
        
        # Load RT-cache dataset
        dataset = RTCacheDataset(
            self.qdrant_client,
            self.collection,
            self.collection_name,
            sample_fraction=self.sample_fraction,
            max_samples=max_samples
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        # Training loop
        self.vae.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            
            for batch in dataloader:
                embeddings = batch["embedding"].to(self.device)
                actions = batch["action"].to(self.device)
                
                # Concatenate embedding and action
                state_action = torch.cat([embeddings, actions], dim=1)
                
                # Forward pass
                recon, mu, logvar, z = self.vae(state_action)
                
                # Compute losses
                recon_loss = F.mse_loss(recon, state_action, reduction='mean')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / state_action.size(0)
                
                total_loss_batch = recon_loss + 0.001 * kl_loss  # β=0.001
                
                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_kl = total_kl_loss / len(dataloader)
            
            if epoch % 10 == 0:
                print(f"[VAE] Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
        
        print("[BehaviorRetrieval] VAE training completed")
    
    def train_bc_head(
        self,
        num_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_samples: int = 50000
    ):
        """
        Train the BC head on VAE embeddings
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size  
            learning_rate: Learning rate
            max_samples: Maximum samples for BC training
        """
        print(f"[BehaviorRetrieval] Training BC head for {num_epochs} epochs...")
        
        # Load dataset for BC training
        dataset = RTCacheDataset(
            self.qdrant_client,
            self.collection,
            self.collection_name,
            sample_fraction=self.sample_fraction,
            max_samples=max_samples
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Optimizer
        optimizer = optim.Adam(self.bc_head.parameters(), lr=learning_rate)
        
        # Training loop
        self.vae.eval()  # Fix VAE weights
        self.bc_head.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                embeddings = batch["embedding"].to(self.device)
                actions = batch["action"].to(self.device)
                
                # Get VAE embeddings (frozen)
                with torch.no_grad():
                    state_action = torch.cat([embeddings, actions], dim=1)
                    vae_embedding = self.vae.get_embedding(state_action)
                
                # Predict actions with BC head
                predicted_actions = self.bc_head(vae_embedding)
                
                # Compute BC loss
                bc_loss = F.mse_loss(predicted_actions, actions)
                
                # Backward pass
                optimizer.zero_grad()
                bc_loss.backward()
                optimizer.step()
                
                total_loss += bc_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0:
                print(f"[BC] Epoch {epoch:3d}: Loss={avg_loss:.4f}")
        
        print("[BehaviorRetrieval] BC head training completed")
        self.is_trained = True
    
    def train(
        self,
        vae_epochs: int = 100,
        bc_epochs: int = 50,
        batch_size: int = 256,
        max_samples: int = 100000
    ):
        """
        Train the full Behavior Retrieval pipeline
        """
        print("[BehaviorRetrieval] Starting full training pipeline...")
        
        # Step 1: Train VAE to re-embed state-action pairs
        self.train_vae(
            num_epochs=vae_epochs,
            batch_size=batch_size,
            max_samples=max_samples
        )
        
        # Step 2: Train BC head on VAE embeddings
        self.train_bc_head(
            num_epochs=bc_epochs,
            batch_size=batch_size,
            max_samples=max_samples//2  # Use less data for BC
        )
        
        print("[BehaviorRetrieval] Full training completed!")
    
    def predict_action(self, observation: Dict) -> np.ndarray:
        """
        Predict action for given observation using Behavior Retrieval
        
        Args:
            observation: Dictionary containing 'image' key with RGB image
            
        Returns:
            7-DOF action vector [x,y,z,roll,pitch,yaw,gripper]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract image
        if "image" in observation:
            image = observation["image"]
        elif "rgb" in observation:
            image = observation["rgb"]
        else:
            raise ValueError("No image found in observation")
        
        # Get 2048-D BYOL embedding (same as VINN)
        byol_embedding = self._get_byol_embedding(image)
        
        # For prediction, we need to get the VAE embedding of the state
        # Since we don't know the action yet, we use a zero action
        zero_action = np.zeros(7, dtype=np.float32)
        state_action = np.concatenate([byol_embedding, zero_action])
        
        # Convert to tensor
        state_action_tensor = torch.from_numpy(state_action).float().unsqueeze(0).to(self.device)
        
        # Get VAE embedding
        self.vae.eval()
        self.bc_head.eval()
        
        with torch.no_grad():
            vae_embedding = self.vae.get_embedding(state_action_tensor)
            predicted_action = self.bc_head(vae_embedding)
        
        return predicted_action.squeeze(0).cpu().numpy()
    
    def __call__(self, observation: Dict) -> np.ndarray:
        """Make policy callable"""
        return self.predict_action(observation)
    
    def save_models(self, save_dir: str):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.vae.state_dict(), os.path.join(save_dir, "vae.pth"))
        torch.save(self.bc_head.state_dict(), os.path.join(save_dir, "bc_head.pth"))
        print(f"[BehaviorRetrieval] Models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models"""
        self.vae.load_state_dict(torch.load(os.path.join(save_dir, "vae.pth"), map_location=self.device))
        self.bc_head.load_state_dict(torch.load(os.path.join(save_dir, "bc_head.pth"), map_location=self.device))
        self.is_trained = True
        print(f"[BehaviorRetrieval] Models loaded from {save_dir}")


def main():
    """
    Example usage of Behavior Retrieval with RT-cache integration
    """
    # Initialize Behavior Retrieval policy
    br_policy = BehaviorRetrievalPolicy(
        qdrant_host="localhost",
        qdrant_port=6333,
        mongo_host="mongodb://localhost:27017/",
        embedding_server="http://localhost:8000/predict",
        collection_name="image_collection",
        sample_fraction=0.25,  # Retrieve ~25% of Open-X
        vae_latent_dim=128
    )
    
    # Train the model
    print("[Main] Training Behavior Retrieval model...")
    br_policy.train(
        vae_epochs=50,  # Reduced for demo
        bc_epochs=25,
        batch_size=128,
        max_samples=10000  # Reduced for demo
    )
    
    # Example prediction
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    observation = {"image": dummy_image}
    
    action = br_policy.predict_action(observation)
    print(f"[BehaviorRetrieval] Predicted action: {action}")
    print(f"[BehaviorRetrieval] Action shape: {action.shape}")
    
    # Save models
    br_policy.save_models("behavior_retrieval_models")


if __name__ == "__main__":
    main()