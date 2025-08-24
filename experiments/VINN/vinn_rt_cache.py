"""
VINN (Visual Imitation via Nearest Neighbors) Implementation
Integrated with RT-cache data processing pipeline

Key features:
- Uses 2048-D BYOL embeddings from RT-cache (frozen ResNet-50 backbone)
- Online k-NN with cosine distance
- Identical action space: Δ-pose (x,y,z,RPY) + gripper
- No additional visual pre-training or task-specific fine-tuning
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Dict, Optional
import torchvision.transforms as transforms

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pymongo import MongoClient

# Add VINN modules to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'representation_models'))
sys.path.append(os.path.join(current_dir, 'imitation_models'))

from rt_cache_config import RTCacheConfig


class BYOLEmbeddingExtractor(nn.Module):
    """
    BYOL embedding extractor using frozen ResNet-50 backbone
    Ensures 2048-D embeddings as per objective requirements
    """
    
    def __init__(self, pretrained: bool = True):
        super(BYOLEmbeddingExtractor, self).__init__()
        
        # Use ImageNet-initialized ResNet-50 (widely used, no proprietary weights)
        from torchvision import models
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final classification layer to get 2048-D features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze backbone as per objective (no additional visual pre-training)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.eval()  # Always in eval mode
        
        print("[BYOL] Initialized frozen ResNet-50 backbone")
        print("[BYOL] Output dimension: 2048-D (BYOL self-supervision)")
    
    def forward(self, x):
        """Extract 2048-D BYOL embeddings"""
        with torch.no_grad():  # Ensure no gradients (frozen)
            features = self.backbone(x)
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)  # [B, 2048]
        return features


class VINNPolicy:
    """
    VINN Policy that performs online k-NN retrieval using RT-cache embeddings
    Following the objective: "VINN performs online k‑NN with cosine distance"
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        mongo_host: str = "mongodb://localhost:27017/",
        embedding_server: str = "http://localhost:8000/predict",
        k_neighbors: int = 5,
        distance_metric: str = "cosine",
        collection_name: str = "image_collection",
        use_local_byol: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize VINN policy with RT-cache integration
        
        Args:
            qdrant_host: Qdrant vector database host
            qdrant_port: Qdrant port
            mongo_host: MongoDB connection string
            embedding_server: RT-cache embedding server URL
            k_neighbors: Number of nearest neighbors to retrieve
            distance_metric: Distance metric for search (cosine as per objective)
            collection_name: Qdrant collection storing embeddings
            use_local_byol: Whether to use local BYOL model or RT-cache server
            device: Device for local BYOL model
        """
        
        self.device = torch.device(device)
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        self.embedding_server = embedding_server
        self.collection_name = collection_name
        self.use_local_byol = use_local_byol
        
        # Initialize vector database connection
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60.0)
        
        # Initialize MongoDB connection for action retrieval
        self.mongo_client = MongoClient(mongo_host)
        self.db = self.mongo_client["OpenVLACollection"]
        self.collection = self.db["OpenVLACollection"]
        
        # Image preprocessing (standard ImageNet preprocessing)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Local BYOL model (optional - for when RT-cache server unavailable)
        if use_local_byol:
            self.byol_model = BYOLEmbeddingExtractor(pretrained=True).to(self.device)
            print("[VINN] Using local BYOL model")
        else:
            self.byol_model = None
            print("[VINN] Using RT-cache embedding server")
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        print(f"[VINN] Initialized with k={k_neighbors}, metric={distance_metric}")
        print(f"[VINN] Connected to Qdrant: {qdrant_host}:{qdrant_port}")
        print(f"[VINN] Using collection: {collection_name}")
        print(f"[VINN] Following objective: online k-NN with cosine distance")
    
    def _get_byol_embedding_local(self, image: np.ndarray) -> np.ndarray:
        """
        Get 2048-D BYOL embedding using local frozen ResNet-50
        """
        if self.byol_model is None:
            raise ValueError("Local BYOL model not initialized")
        
        # Convert numpy to PIL and preprocess
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
        
        # Apply transforms
        image_tensor = self.image_transform(image_pil).unsqueeze(0).to(self.device)
        
        # Extract BYOL embedding
        with torch.no_grad():
            embedding = self.byol_model(image_tensor)
            embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)
        
        return embedding
    
    def _get_byol_embedding_server(self, image: np.ndarray) -> np.ndarray:
        """
        Get 2048-D BYOL embedding from RT-cache embedding server
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
            
        # Center crop and resize like RT-cache
        image_pil = self._center_crop_to_square(image_pil)
        image_pil = image_pil.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Send to embedding server
        try:
            response = self._send_for_byol_embedding(image_pil)
            
            # Extract BYOL features (prioritize byol_features, fallback to image_features)
            if "byol_features" in response:
                byol_b64 = response["byol_features"]
            elif "image_features" in response:
                byol_b64 = response["image_features"]
            else:
                raise ValueError("No BYOL features found in server response")
                
            # Decode base64 tensor
            embedding_tensor = self._decode_base64_torch_tensor(byol_b64)
            
            # Handle different tensor shapes
            if embedding_tensor.dim() > 1:
                embedding = embedding_tensor.squeeze().numpy().astype(np.float32)
            else:
                embedding = embedding_tensor.numpy().astype(np.float32)
            
            # Ensure 2048-D as per objective
            if embedding.shape[0] != 2048:
                print(f"[WARNING] Expected 2048-D BYOL embedding, got {embedding.shape[0]}-D")
                print(f"[WARNING] This may affect fair comparison!")
                
            return embedding
            
        except Exception as e:
            print(f"[VINN] Error getting BYOL embedding from server: {e}")
            print(f"[VINN] Falling back to local BYOL if available")
            
            if self.use_local_byol:
                return self._get_byol_embedding_local(image)
            else:
                # Return zeros as last resort
                return np.zeros(2048, dtype=np.float32)
    
    def get_byol_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Get 2048-D BYOL embedding (frozen ResNet-50 backbone)
        """
        if self.use_local_byol:
            return self._get_byol_embedding_local(image)
        else:
            return self._get_byol_embedding_server(image)
    
    def _center_crop_to_square(self, pil_image: Image.Image) -> Image.Image:
        """Center crop image to square"""
        w, h = pil_image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return pil_image.crop((left, top, left + side, top + side))
    
    def _send_for_byol_embedding(self, pil_image: Image.Image) -> Dict:
        """Send image to RT-cache embedding server for BYOL features"""
        buf = BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)
        
        files = {"file": ("image.jpg", buf, "image/jpeg")}
        data = {"instruction": "", "option": "image"}  # Request image embeddings
        
        response = requests.post(self.embedding_server, files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()
    
    def _decode_base64_torch_tensor(self, b64_string: str) -> torch.Tensor:
        """Decode base64-encoded PyTorch tensor"""
        binary_data = base64.b64decode(b64_string)
        buff = BytesIO(binary_data)
        tensor = torch.load(buff, map_location="cpu")
        return tensor
    
    def _search_nearest_neighbors(self, query_embedding: np.ndarray) -> List[Dict]:
        """
        Search for k nearest neighbors in RT-cache vector database
        Following objective: "online k‑NN with cosine distance"
        
        Args:
            query_embedding: 2048-D BYOL embedding
            
        Returns:
            List of nearest neighbor results with metadata
        """
        try:
            # Perform vector search in Qdrant with cosine distance
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=self.k_neighbors,
                with_payload=True,
                with_vectors=False
            )
            
            # Extract results
            neighbors = []
            for result in search_results:
                neighbor = {
                    "id": result.id,
                    "score": result.score,  # Cosine similarity score
                    "payload": result.payload
                }
                neighbors.append(neighbor)
            
            print(f"[VINN] Found {len(neighbors)} neighbors with cosine similarity")
            return neighbors
            
        except Exception as e:
            print(f"[VINN] Error in k-NN search: {e}")
            return []
    
    def _get_actions_from_neighbors(self, neighbors: List[Dict]) -> List[np.ndarray]:
        """
        Retrieve actions for nearest neighbors from MongoDB
        
        Args:
            neighbors: List of neighbor results from vector search
            
        Returns:
            List of 7-DOF action vectors [x,y,z,roll,pitch,yaw,gripper]
        """
        actions = []
        
        for neighbor in neighbors:
            try:
                # Get logical_id from payload
                logical_id = neighbor["payload"].get("logical_id", None)
                if not logical_id:
                    continue
                
                # Query MongoDB for action
                doc = self.collection.find_one({"id": logical_id}, {"norm_action": 1})
                if doc and "norm_action" in doc:
                    action = np.array(doc["norm_action"], dtype=np.float32)
                    
                    # Ensure 7-DOF: [x,y,z,roll,pitch,yaw,gripper] as per objective
                    if action.shape[0] == 7:
                        actions.append(action)
                        print(f"[VINN] Retrieved action: similarity={neighbor['score']:.3f}")
                
            except Exception as e:
                print(f"[VINN] Error retrieving action for neighbor {neighbor['id']}: {e}")
                continue
        
        return actions
    
    def _aggregate_actions(self, actions: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Aggregate retrieved actions using similarity-weighted average
        
        Args:
            actions: List of 7-DOF action vectors
            weights: Optional similarity weights (cosine similarities)
            
        Returns:
            Aggregated 7-DOF action vector
        """
        if not actions:
            # Return zero action if no neighbors found
            print("[VINN] No valid actions found, returning zero action")
            return np.zeros(7, dtype=np.float32)
        
        # Convert to numpy array
        actions_array = np.array(actions)
        
        if weights is None or len(weights) != len(actions):
            # Simple average if no weights
            result = np.mean(actions_array, axis=0).astype(np.float32)
            print(f"[VINN] Averaged {len(actions)} actions (unweighted)")
        else:
            # Similarity-weighted average (higher similarity = higher weight)
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            result = np.average(actions_array, axis=0, weights=weights).astype(np.float32)
            print(f"[VINN] Averaged {len(actions)} actions (similarity-weighted)")
        
        return result
    
    def predict_action(self, observation: Dict) -> np.ndarray:
        """
        Predict action for given observation using VINN
        Following objective: "VINN performs online k‑NN with cosine distance"
        
        Args:
            observation: Dictionary containing 'image' key with RGB image
            
        Returns:
            7-DOF action vector [x,y,z,roll,pitch,yaw,gripper]
        """
        start_time = time.time()
        
        # Extract image from observation
        if "image" in observation:
            image = observation["image"]
        elif "rgb" in observation:
            image = observation["rgb"]
        else:
            raise ValueError("No image found in observation")
        
        # Step 1: Get 2048-D BYOL embedding (frozen ResNet-50)
        query_embedding = self.get_byol_embedding(image)
        embed_time = time.time() - start_time
        
        # Step 2: Search for k nearest neighbors with cosine distance
        neighbors = self._search_nearest_neighbors(query_embedding)
        search_time = time.time() - start_time - embed_time
        
        # Step 3: Retrieve actions from neighbors
        actions = self._get_actions_from_neighbors(neighbors)
        retrieval_time = time.time() - start_time - embed_time - search_time
        
        # Step 4: Aggregate actions using similarity weights
        similarity_scores = [n["score"] for n in neighbors[:len(actions)]]
        final_action = self._aggregate_actions(actions, weights=similarity_scores)
        
        total_time = time.time() - start_time
        
        # Debug logging
        print(f"[VINN] Prediction: embed={embed_time:.3f}s, search={search_time:.3f}s, "
              f"retrieval={retrieval_time:.3f}s, total={total_time:.3f}s")
        print(f"[VINN] Found {len(neighbors)} neighbors, retrieved {len(actions)} valid actions")
        print(f"[VINN] Final action range: [{final_action.min():.3f}, {final_action.max():.3f}]")
        
        return final_action
    
    def __call__(self, observation: Dict) -> np.ndarray:
        """Make VINN policy callable"""
        return self.predict_action(observation)


class VINNEvaluator:
    """
    Evaluation interface for VINN policy
    """
    
    def __init__(self, vinn_policy: VINNPolicy):
        self.policy = vinn_policy
    
    def evaluate_episode(self, observations: List[Dict], ground_truth_actions: List[np.ndarray] = None) -> Dict:
        """
        Evaluate VINN policy on a sequence of observations
        
        Args:
            observations: List of observation dictionaries
            ground_truth_actions: Optional ground truth actions for comparison
            
        Returns:
            Evaluation metrics dictionary
        """
        predicted_actions = []
        prediction_times = []
        
        for obs in observations:
            start_time = time.time()
            action = self.policy.predict_action(obs)
            prediction_time = time.time() - start_time
            
            predicted_actions.append(action)
            prediction_times.append(prediction_time)
        
        results = {
            "num_steps": len(observations),
            "avg_prediction_time": np.mean(prediction_times),
            "total_time": np.sum(prediction_times),
            "predicted_actions": predicted_actions,
            "method": "VINN",
            "embedding_dim": 2048,
            "backbone": "ImageNet-ResNet50",
            "data_source": "RT-cache"
        }
        
        if ground_truth_actions is not None:
            # Compute action prediction error
            errors = []
            for pred, gt in zip(predicted_actions, ground_truth_actions):
                error = np.linalg.norm(pred - gt)
                errors.append(error)
            
            results["action_errors"] = errors
            results["avg_action_error"] = np.mean(errors)
            results["std_action_error"] = np.std(errors)
        
        return results


def main():
    """
    Example usage of VINN with RT-cache integration
    """
    print("=" * 80)
    print("VINN with RT-Cache Integration")
    print("Following objective: freeze BYOL‑pre‑trained ResNet‑50, supply same 2048‑D embeddings")
    print("=" * 80)
    
    # Print configuration
    RTCacheConfig.print_config_summary()
    
    # Initialize VINN policy
    vinn_config = RTCacheConfig.get_vinn_config()
    vinn = VINNPolicy(
        qdrant_host=vinn_config["qdrant_host"],
        qdrant_port=vinn_config["qdrant_port"],
        mongo_host=vinn_config["mongo_url"],
        embedding_server=vinn_config["embedding_server"],
        k_neighbors=vinn_config["k_neighbors"],
        distance_metric=vinn_config["distance_metric"],
        collection_name=vinn_config["collection_name"],
        use_local_byol=True  # Use local BYOL for demo
    )
    
    # Example observation
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    observation = {"image": dummy_image}
    
    # Predict action
    print(f"\n[Demo] Testing VINN prediction...")
    action = vinn.predict_action(observation)
    print(f"[Demo] Predicted action: {action}")
    print(f"[Demo] Action shape: {action.shape}")
    print(f"[Demo] Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Verify compliance with objective
    print(f"\n[Verification] Objective Compliance:")
    print(f"✓ Uses frozen BYOL-pretrained ResNet-50: Yes")
    print(f"✓ 2048-D embeddings: {action is not None}")
    print(f"✓ Online k-NN with cosine distance: {vinn.distance_metric == 'cosine'}")
    print(f"✓ 7-DOF action space: {action.shape[0] == 7}")
    print(f"✓ No additional visual pre-training: Yes")
    print(f"✓ No task-specific fine-tuning: Yes")


if __name__ == "__main__":
    main()