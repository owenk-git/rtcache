#!/usr/bin/env python3
"""
PROPER VINN Implementation with BYOL Fine-tuning
Matches the original paper's two-phase approach exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import time
from tqdm import tqdm

# Try to import BYOL (install with: pip install byol-pytorch)
try:
    from byol_pytorch import BYOL
    BYOL_AVAILABLE = True
except ImportError:
    print("⚠️  BYOL not available. Install with: pip install byol-pytorch")
    BYOL_AVAILABLE = False

################################################################################
#                    PHASE 1: BYOL Self-Supervised Training
################################################################################

class BYOLTrainer:
    """BYOL trainer for learning visual representations on demonstration data"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        if not BYOL_AVAILABLE:
            raise ImportError("BYOL not available. Install with: pip install byol-pytorch")
        
        # Initialize ResNet-50 backbone (ImageNet pretrained)
        self.backbone = resnet50(pretrained=True)
        
        # BYOL wrapper around the backbone
        self.byol = BYOL(
            self.backbone,
            image_size=224,
            hidden_layer='avgpool',  # Use avgpool layer for 2048-D features
            projection_size=256,
            projection_hidden_size=4096,
            moving_average_decay=0.99
        ).to(device)
        
        # Optimizer for BYOL training
        self.optimizer = torch.optim.Adam(self.byol.parameters(), lr=3e-4)
        
        # Image transforms for BYOL (with augmentations)
        self.byol_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ BYOL trainer initialized with ImageNet-pretrained ResNet-50")
    
    def train_byol(self, image_dataset, epochs=100, batch_size=32):
        """
        Train BYOL on demonstration images (Phase 1 of VINN)
        Following Section III.A of the paper
        """
        print(f"\n🔧 PHASE 1: BYOL Self-Supervised Training")
        print(f"Training on {len(image_dataset)} demonstration images for {epochs} epochs")
        
        dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        self.byol.train()
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for images in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.byol(images)
                loss.backward()
                self.optimizer.step()
                
                # Update target network (EMA)
                self.byol.update_moving_average()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}: BYOL Loss = {avg_loss:.6f}")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_model(f"./byol_checkpoints/byol_epoch_{epoch+1}.pth")
        
        print("✅ BYOL training completed!")
        return self.get_encoder()
    
    def get_encoder(self):
        """Extract the trained encoder from BYOL"""
        # Get the online network backbone
        encoder = self.byol.online_encoder.net
        
        # Modify to output 2048-D features (remove final projection layers)
        if hasattr(encoder, 'fc'):
            encoder.fc = nn.Identity()
        
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
            
        return encoder
    
    def save_model(self, path):
        """Save BYOL model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'byol_state_dict': self.byol.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved BYOL model to {path}")
    
    def load_model(self, path):
        """Load BYOL model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.byol.load_state_dict(checkpoint['byol_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded BYOL model from {path}")

################################################################################
#                    PHASE 2: k-NN Based Action Prediction
################################################################################

class ProperVINN:
    """
    PROPER VINN Implementation following the original paper exactly
    Two-phase approach: BYOL training → k-NN inference
    """
    
    def __init__(self, device='cuda', k=16):
        self.device = device
        self.k = k
        
        # Phase 1 components
        self.byol_trainer = None
        self.encoder = None
        
        # Phase 2 components  
        self.database_embeddings = None
        self.database_actions = None
        
        # Image preprocessing for inference (no augmentation)
        self.inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ PROPER VINN initialized with k={k}")
    
    def phase1_train_visual_representation(self, image_dataset, epochs=100):
        """
        PHASE 1: Visual Representation Learning (Section III.A)
        Train BYOL on demonstration images
        """
        if not BYOL_AVAILABLE:
            print("⚠️  BYOL not available, using ImageNet features only")
            self.encoder = resnet50(pretrained=True)
            self.encoder.fc = nn.Identity()
            self.encoder = self.encoder.to(self.device)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            return self.encoder
        
        self.byol_trainer = BYOLTrainer(device=self.device)
        self.encoder = self.byol_trainer.train_byol(image_dataset, epochs=epochs)
        
        print("✅ Phase 1 Complete: Visual representation learned")
        return self.encoder
    
    def phase2_build_demonstration_database(self, demo_dataset):
        """
        Build database of demonstration embeddings and actions
        """
        print(f"\n🔍 PHASE 2: Building demonstration database")
        print(f"Processing {len(demo_dataset)} demonstrations...")
        
        embeddings = []
        actions = []
        
        self.encoder.eval()
        with torch.no_grad():
            for i in range(len(demo_dataset)):
                image, action = demo_dataset[i]
                
                # Process image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                image_tensor = self.inference_transform(image).unsqueeze(0).to(self.device)
                
                # Extract 2048-D embedding
                embedding = self.encoder(image_tensor).cpu().numpy()[0]
                embeddings.append(embedding)
                actions.append(action)
        
        self.database_embeddings = np.array(embeddings)
        self.database_actions = np.array(actions)
        
        print(f"✅ Database built: {len(self.database_embeddings)} demonstrations")
        print(f"✅ Embedding dimension: {self.database_embeddings.shape[1]}")
    
    def euclidean_distance(self, x, y):
        """Euclidean distance as per paper: ||e - e^(i)||2"""
        return np.linalg.norm(x - y)
    
    def predict_action(self, query_image):
        """
        PHASE 2: k-NN Based Action Prediction (Section III.B)
        Following the exact formula from the paper
        """
        # Extract query embedding
        if isinstance(query_image, np.ndarray):
            query_image = Image.fromarray(query_image)
        query_tensor = self.inference_transform(query_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.encoder(query_tensor).cpu().numpy()[0]
        
        # Compute distances to all demonstrations
        distances = []
        for i, demo_embedding in enumerate(self.database_embeddings):
            dist = self.euclidean_distance(query_embedding, demo_embedding)
            distances.append((dist, i))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[0])
        top_k = distances[:self.k]
        
        # Apply Euclidean kernel weighting: exp(-||e - e^(i)||2)
        weights = []
        neighbor_actions = []
        
        for dist, idx in top_k:
            weight = np.exp(-dist)  # Euclidean kernel
            weights.append(weight)
            neighbor_actions.append(self.database_actions[idx])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average: â = Σ w_i * a_i
        weighted_action = np.zeros_like(neighbor_actions[0])
        for weight, action in zip(weights, neighbor_actions):
            weighted_action += weight * action
        
        return weighted_action
    
    def save_model(self, save_dir):
        """Save trained VINN model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save encoder
        torch.save(self.encoder.state_dict(), f"{save_dir}/encoder.pth")
        
        # Save database
        np.save(f"{save_dir}/database_embeddings.npy", self.database_embeddings)
        np.save(f"{save_dir}/database_actions.npy", self.database_actions)
        
        print(f"✅ VINN model saved to {save_dir}")

################################################################################
#                               Dataset Classes
################################################################################

class ImageOnlyDataset(Dataset):
    """Dataset for BYOL training (images only, no actions)"""
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image

class DemonstrationDataset(Dataset):
    """Dataset for demonstration database (images + actions)"""
    def __init__(self, images, actions):
        self.images = images
        self.actions = actions
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.actions[idx]

################################################################################
#                               Demo Usage
################################################################################

def demo_proper_vinn():
    """Demonstrate proper VINN training and inference"""
    print("🚀 PROPER VINN Demo - Two Phase Training")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate dummy data for demonstration
    print("Generating dummy demonstration data...")
    n_demos = 1000
    dummy_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(n_demos)]
    dummy_actions = [np.random.randn(7) for _ in range(n_demos)]
    
    # Initialize PROPER VINN
    vinn = ProperVINN(device=device, k=16)
    
    # Phase 1: BYOL Visual Representation Learning
    print("\n" + "="*60)
    print("PHASE 1: BYOL Self-Supervised Learning")
    print("="*60)
    
    image_dataset = ImageOnlyDataset(
        dummy_images,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    
    encoder = vinn.phase1_train_visual_representation(image_dataset, epochs=5)  # Small epochs for demo
    
    # Phase 2: Build Demonstration Database
    print("\n" + "="*60)
    print("PHASE 2: Build Demonstration Database")
    print("="*60)
    
    demo_dataset = DemonstrationDataset(dummy_images, dummy_actions)
    vinn.phase2_build_demonstration_database(demo_dataset)
    
    # Test inference
    print("\n" + "="*60)
    print("TESTING k-NN INFERENCE")
    print("="*60)
    
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    start_time = time.time()
    predicted_action = vinn.predict_action(test_image)
    inference_time = time.time() - start_time
    
    print(f"✅ Predicted action: {predicted_action}")
    print(f"✅ Inference time: {inference_time*1000:.2f}ms")
    
    # Save model
    vinn.save_model("./proper_vinn_models")
    print("✅ PROPER VINN training completed!")

if __name__ == "__main__":
    demo_proper_vinn()