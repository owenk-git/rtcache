"""
Common utilities for making embedding requests.

Extracted from repeated embedding server requests across scripts.
"""

import requests
import torch
from io import BytesIO
from PIL import Image
from typing import Dict, Optional, Any
import logging


class EmbeddingClient:
    """Simple client for embedding server requests."""
    
    def __init__(self, server_url: str, timeout: int = 300):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def get_image_embedding(self, 
                          image: Image.Image, 
                          embedding_type: str = "both") -> Dict[str, torch.Tensor]:
        """
        Get embedding for image.
        
        Args:
            image: PIL Image
            embedding_type: "image", "both", "clip", "openvla"
            
        Returns:
            Dict with embedding tensors
        """
        try:
            # Prepare request
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            
            files = {"file": ("image.jpg", buffer, "image/jpeg")}
            data = {"instruction": "", "option": embedding_type}
            
            # Make request
            response = requests.post(
                f"{self.server_url}/predict",
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Extract embeddings based on type
            embeddings = {}
            
            if "image_features" in response_data:
                embeddings["openvla_image"] = self._decode_tensor(response_data["image_features"])
                
            if "clip_image_features" in response_data:
                embeddings["clip_image"] = self._decode_tensor(response_data["clip_image_features"])
                
            if "llm_features" in response_data:
                embeddings["openvla_text"] = self._decode_tensor(response_data["llm_features"])
                
            if "clip_text_features" in response_data:
                embeddings["clip_text"] = self._decode_tensor(response_data["clip_text_features"])
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding request failed: {str(e)}")
            raise
    
    def get_text_embedding(self, text: str) -> Dict[str, torch.Tensor]:
        """Get embedding for text only."""
        try:
            data = {"instruction": text, "option": "text"}
            
            response = requests.post(
                f"{self.server_url}/predict",
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            embeddings = {}
            if "llm_features" in response_data:
                embeddings["openvla_text"] = self._decode_tensor(response_data["llm_features"])
                
            if "clip_text_features" in response_data:
                embeddings["clip_text"] = self._decode_tensor(response_data["clip_text_features"])
                
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Text embedding request failed: {str(e)}")
            raise
    
    def get_multimodal_embedding(self, 
                                image: Image.Image, 
                                text: str) -> Dict[str, torch.Tensor]:
        """Get embeddings for both image and text."""
        try:
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            
            files = {"file": ("image.jpg", buffer, "image/jpeg")}
            data = {"instruction": text, "option": "both"}
            
            response = requests.post(
                f"{self.server_url}/predict",
                files=files,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            embeddings = {}
            
            # Image embeddings
            if "image_features" in response_data:
                embeddings["openvla_image"] = self._decode_tensor(response_data["image_features"])
                
            if "clip_image_features" in response_data:
                embeddings["clip_image"] = self._decode_tensor(response_data["clip_image_features"])
                
            # Text embeddings
            if "llm_features" in response_data:
                embeddings["openvla_text"] = self._decode_tensor(response_data["llm_features"])
                
            if "clip_text_features" in response_data:
                embeddings["clip_text"] = self._decode_tensor(response_data["clip_text_features"])
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Multimodal embedding request failed: {str(e)}")
            raise
    
    def _decode_tensor(self, b64_string: str) -> torch.Tensor:
        """Decode base64 encoded tensor."""
        import base64
        bin_data = base64.b64decode(b64_string)
        buffer = BytesIO(bin_data)
        tensor = torch.load(buffer, map_location="cpu")
        return tensor
    
    def health_check(self) -> bool:
        """Check if embedding server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False


def create_embedding_client(config) -> EmbeddingClient:
    """Simple factory for embedding client."""
    return EmbeddingClient(
        server_url=config.server.embedding_url,
        timeout=getattr(config.server, 'embedding_timeout', 300)
    )