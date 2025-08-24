"""
ML model implementations for retrieval system.

Extracted from retrieval_server.py to separate model logic from server logic.
"""

import os
import logging
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image


class VINNPredictor:
    """
    Visual Imitation via NeRF Networks predictor.
    
    Extracted from retrieval_server.py for modularity.
    """
    
    def __init__(self, model_dir: str, device: str = "cuda", k: int = 16):
        """
        Initialize VINN predictor.
        
        Args:
            model_dir: Directory containing VINN model files
            device: Device to run model on
            k: Number of nearest neighbors to use
        """
        self.model_dir = model_dir
        self.device = device
        self.k = k
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
        self.encoder = None
        self.database_embeddings = None
        self.database_actions = None
        self.transform = None
        
    def initialize(self) -> bool:
        """
        Load VINN model and database.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"Loading VINN model from {self.model_dir}...")
            
            # Load encoder
            self.encoder = resnet50(pretrained=False)
            self.encoder.fc = nn.Identity()
            
            encoder_path = os.path.join(self.model_dir, "encoder.pth")
            if not os.path.exists(encoder_path):
                self.logger.warning(f"VINN encoder not found at {encoder_path}")
                return False
                
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.encoder = self.encoder.to(self.device)
            self.encoder.eval()
            
            # Load database
            embeddings_path = os.path.join(self.model_dir, "database_embeddings.npy")
            actions_path = os.path.join(self.model_dir, "database_actions.npy")
            
            if not os.path.exists(embeddings_path) or not os.path.exists(actions_path):
                self.logger.warning(f"VINN database files not found in {self.model_dir}")
                return False
                
            self.database_embeddings = np.load(embeddings_path)
            self.database_actions = np.load(actions_path)
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.initialized = True
            self.logger.info(f"VINN loaded successfully: {len(self.database_embeddings)} demonstrations")
            return True
            
        except Exception as e:
            self.logger.error(f"VINN initialization failed: {e}")
            self.initialized = False
            return False
    
    def predict(self, image: Image.Image) -> Optional[List[float]]:
        """
        Predict action for given image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Predicted action as list of floats, or None if prediction fails
        """
        if not self.initialized:
            self.logger.warning("VINN predictor not initialized")
            return None
        
        try:
            # Preprocess image
            query_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get query embedding
            with torch.no_grad():
                query_embedding = self.encoder(query_tensor).cpu().numpy()[0]
            
            # Find k nearest neighbors
            distances = []
            for i, demo_embedding in enumerate(self.database_embeddings):
                dist = np.linalg.norm(query_embedding - demo_embedding)
                distances.append((dist, i))
            
            # Sort and get top k
            distances.sort(key=lambda x: x[0])
            top_k = distances[:self.k]
            
            # Weighted average of actions
            weights = []
            actions = []
            
            for dist, idx in top_k:
                weight = np.exp(-dist)
                weights.append(weight)
                actions.append(self.database_actions[idx])
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Compute weighted average action
            predicted_action = np.zeros_like(actions[0])
            for weight, action in zip(weights, actions):
                predicted_action += weight * action
            
            return predicted_action.tolist()
            
        except Exception as e:
            self.logger.error(f"VINN prediction failed: {e}")
            return None


class BehaviorRetrievalPredictor:
    """
    Behavior Retrieval predictor.
    
    Extracted from retrieval_server.py for modularity.
    """
    
    def __init__(self, model_dir: str, device: str = "cuda"):
        """
        Initialize BehaviorRetrieval predictor.
        
        Args:
            model_dir: Directory containing BR model files
            device: Device to run model on
        """
        self.model_dir = model_dir
        self.device = device
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
        self.visual_encoder = None
        self.policy = None
        
    def initialize(self) -> bool:
        """
        Load BehaviorRetrieval model.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"Loading BehaviorRetrieval model from {self.model_dir}...")
            
            # Create model architecture
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 64)
            ).to(self.device)
            
            self.policy = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 7)
            ).to(self.device)
            
            # Load weights
            policy_path = os.path.join(self.model_dir, "policy_target_training.pth")
            if not os.path.exists(policy_path):
                self.logger.warning(f"BR policy not found at {policy_path}")
                return False
                
            self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            
            # Set to eval mode
            self.visual_encoder.eval()
            self.policy.eval()
            
            self.initialized = True
            self.logger.info("BehaviorRetrieval model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"BehaviorRetrieval initialization failed: {e}")
            self.initialized = False
            return False
    
    def predict(self, image: Image.Image) -> Optional[List[float]]:
        """
        Predict action for given image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Predicted action as list of floats, or None if prediction fails
        """
        if not self.initialized:
            self.logger.warning("BehaviorRetrieval predictor not initialized")
            return None
        
        try:
            # Preprocess image (84x84 for BR)
            image_resized = image.resize((84, 84), Image.LANCZOS)
            image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                visual_features = self.visual_encoder(image_tensor)
                predicted_action = self.policy(visual_features).cpu().numpy()[0]
            
            return predicted_action.tolist()
            
        except Exception as e:
            self.logger.error(f"BehaviorRetrieval prediction failed: {e}")
            return None


def create_predictors(config, models_to_load: List[str]) -> dict:
    """
    Factory function to create and initialize predictors.
    
    Args:
        config: Configuration object
        models_to_load: List of model names to load ('vinn', 'br')
        
    Returns:
        Dictionary mapping model names to initialized predictors
    """
    predictors = {}
    
    if 'vinn' in models_to_load:
        vinn = VINNPredictor(
            model_dir=getattr(config.paths, 'vinn_model_dir', '../VINN/vinn_target_models'),
            device=config.model.device
        )
        if vinn.initialize():
            predictors['vinn'] = vinn
    
    if 'br' in models_to_load:
        br = BehaviorRetrievalPredictor(
            model_dir=getattr(config.paths, 'br_model_dir', '../BehaviorRetrieval/br_target_models'),
            device=config.model.device
        )
        if br.initialize():
            predictors['br'] = br
    
    return predictors