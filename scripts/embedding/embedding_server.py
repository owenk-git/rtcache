#!/usr/bin/env python3
"""
Embedding Server for RT-Cache System

This FastAPI server generates vision-language embeddings using OpenVLA and CLIP models.
It provides endpoints for processing images and text instructions to create embeddings
used for action trajectory retrieval.

Author: RT-Cache Team
Date: 2024
"""

import os
import sys
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO
import time

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor
)
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "config"))

# Load centralized configuration
from rt_cache_config import get_config

def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """Server configuration using centralized config"""
    
    def __init__(self):
        config = get_config()
        
        # Server settings
        self.host = config.server.embedding_host
        self.port = config.server.embedding_port
        self.workers = config.server.embedding_workers
        
        # Model settings
        self.device = config.model.device
        self.dtype = config.model.model_dtype
        self.use_flash_attention = config.model.use_flash_attention
        
        # Processing settings
        self.max_batch_size = config.model.model_batch_size
        self.image_size = (224, 224)
        
        # Logging
        self.log_level = config.paths.log_level

# ============================================================================
# Response Models
# ============================================================================

class EmbeddingResponse(BaseModel):
    """Response model for embedding endpoint"""
    
    # OpenVLA embeddings
    image_features: Optional[str] = None  # Base64 encoded tensor
    llm_features: Optional[str] = None    # Base64 encoded tensor
    
    # CLIP embeddings
    clip_image_features: Optional[str] = None  # Base64 encoded tensor
    clip_text_features: Optional[str] = None   # Base64 encoded tensor
    
    # Metadata
    processing_time: float
    model_versions: Dict[str, str]

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    
    status: str
    models_loaded: Dict[str, bool]
    device: str
    memory_usage: Dict[str, float]

# ============================================================================
# Embedding Server
# ============================================================================

class EmbeddingServer:
    """
    FastAPI server for generating vision-language embeddings.
    
    This server provides:
    - OpenVLA embeddings (DINO + SigLIP for images, LLM for text)
    - CLIP embeddings (ViT for images, text encoder for text)
    - Health monitoring and metrics
    """
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the embedding server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="RT-Cache Embedding Server",
            description="Vision-Language embedding generation for robot control",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self._setup_cors()
        
        # Load models
        self._load_models()
        
        # Setup routes
        self._setup_routes()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_images": 0,
            "total_text": 0,
            "avg_processing_time": 0
        }
        
    def _setup_cors(self):
        """Configure CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        
    def _load_models(self):
        """Load OpenVLA and CLIP models"""
        self.logger.info("Loading models...")
        
        try:
            # Load OpenVLA
            self.logger.info("Loading OpenVLA model...")
            self.openvla_processor = AutoProcessor.from_pretrained(
                "openvla/openvla-7b",
                trust_remote_code=True
            )
            
            # Determine dtype
            if self.config.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif self.config.dtype == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
                
            self.openvla_model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                attn_implementation="flash_attention_2" if self.config.use_flash_attention else "eager",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.config.device)
            
            self.logger.info("OpenVLA model loaded successfully")
            
            # Load CLIP
            self.logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.config.device)
            
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            
            self.logger.info("CLIP model loaded successfully")
            
            # Model info
            self.model_info = {
                "openvla": "openvla/openvla-7b",
                "clip": "openai/clip-vit-base-patch32",
                "device": self.config.device,
                "dtype": str(torch_dtype)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
            
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "service": "RT-Cache Embedding Server",
                "version": "1.0.0",
                "status": "running"
            }
            
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return self._get_health_status()
            
        @self.app.post("/predict", response_model=EmbeddingResponse)
        async def generate_embeddings(
            file: Optional[UploadFile] = File(None),
            instruction: str = Form(""),
            option: str = Form("both")
        ):
            """
            Generate embeddings for image and/or text.
            
            Args:
                file: Optional image file
                instruction: Text instruction
                option: Processing option ("image", "text", or "both")
                
            Returns:
                EmbeddingResponse with base64-encoded embeddings
            """
            return await self._process_embedding_request(file, instruction, option)
            
        @self.app.get("/stats")
        async def get_statistics():
            """Get server statistics"""
            return self.stats
            
    async def _process_embedding_request(
        self,
        file: Optional[UploadFile],
        instruction: str,
        option: str
    ) -> EmbeddingResponse:
        """
        Process embedding generation request.
        
        Args:
            file: Image file
            instruction: Text instruction
            option: Processing option
            
        Returns:
            EmbeddingResponse with embeddings
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            result = {}
            
            # Load or create dummy image
            if file is not None:
                image = await self._load_image(file)
                self.stats["total_images"] += 1
            else:
                image = self._get_dummy_image()
                
            # Prepare prompt for OpenVLA
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            
            # Generate OpenVLA embeddings
            if option in ["image", "both"] and file is not None:
                openvla_embeddings = self._generate_openvla_embeddings(image, prompt)
                result.update(openvla_embeddings)
                
            # Generate CLIP embeddings
            if option in ["image", "both"] and file is not None:
                clip_image_emb = self._generate_clip_image_embedding(image)
                result["clip_image_features"] = clip_image_emb
                
            if option in ["text", "both"] and instruction:
                clip_text_emb = self._generate_clip_text_embedding(instruction)
                result["clip_text_features"] = clip_text_emb
                self.stats["total_text"] += 1
                
            # Generate OpenVLA text embeddings
            if option in ["text", "both"] and instruction:
                llm_features = self._generate_openvla_text_embedding(prompt)
                result["llm_features"] = llm_features
                
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update average processing time
            n = self.stats["total_requests"]
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (n - 1) + processing_time) / n
            )
            
            return EmbeddingResponse(
                **result,
                processing_time=processing_time,
                model_versions=self.model_info
            )
            
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def _generate_openvla_embeddings(self, image: Image.Image, prompt: str) -> Dict[str, str]:
        """
        Generate OpenVLA image embeddings.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            
        Returns:
            Dictionary with base64-encoded embeddings
        """
        # Process inputs
        inputs = self.openvla_processor(
            prompt,
            image,
            return_tensors="pt"
        ).to(self.config.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # Extract pixel values
            pixel_values = inputs.pixel_values
            
            # Handle 6-channel input (DINO + SigLIP)
            if pixel_values.shape[1] == 6:
                dino_input = pixel_values[:, :3, :, :]
                siglip_input = pixel_values[:, 3:, :, :]
            else:
                dino_input = pixel_values
                siglip_input = pixel_values
                
            # Generate DINO features
            dino_features = self.openvla_model.vision_backbone.featurizer(dino_input)
            final_dino_features = dino_features[:, -1, :]  # Last token, shape [1, 1024]
            
            # Generate SigLIP features
            siglip_features = self.openvla_model.vision_backbone.fused_featurizer(siglip_input)
            final_siglip_features = siglip_features.mean(dim=1)  # Average pooling, shape [1, 1152]
            
            # Concatenate features
            concatenated_features = torch.cat(
                (final_dino_features, final_siglip_features),
                dim=-1
            )  # Shape [1, 2176]
            
        # Encode to base64
        image_features_b64 = self._encode_tensor_to_base64(concatenated_features)
        
        return {"image_features": image_features_b64}
        
    def _generate_openvla_text_embedding(self, prompt: str) -> str:
        """
        Generate OpenVLA text (LLM) embeddings.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Base64-encoded text embedding
        """
        # Create dummy image for text-only processing
        dummy_image = self._get_dummy_image()
        
        # Process inputs
        inputs = self.openvla_processor(
            prompt,
            dummy_image,
            return_tensors="pt"
        ).to(self.config.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # Get LLM embeddings
            outputs = self.openvla_model.language_model.model.embed_tokens(
                inputs.input_ids
            )
            # Average pooling over sequence
            llm_features = outputs.mean(dim=1)  # Shape [1, 4096]
            
        # Encode to base64
        return self._encode_tensor_to_base64(llm_features)
        
    def _generate_clip_image_embedding(self, image: Image.Image) -> str:
        """
        Generate CLIP image embedding.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64-encoded CLIP image embedding
        """
        # Process image
        clip_inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.config.device)
        
        with torch.no_grad():
            clip_image_emb = self.clip_model.get_image_features(**clip_inputs)
            
        return self._encode_tensor_to_base64(clip_image_emb)
        
    def _generate_clip_text_embedding(self, text: str) -> str:
        """
        Generate CLIP text embedding.
        
        Args:
            text: Text string
            
        Returns:
            Base64-encoded CLIP text embedding
        """
        # Process text
        clip_inputs = self.clip_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.config.device)
        
        with torch.no_grad():
            clip_text_emb = self.clip_model.get_text_features(**clip_inputs)
            
        return self._encode_tensor_to_base64(clip_text_emb)
        
    def _encode_tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """
        Encode PyTorch tensor to base64 string.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Base64-encoded string
        """
        buffer = BytesIO()
        torch.save(tensor.cpu(), buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
        
    async def _load_image(self, file: UploadFile) -> Image.Image:
        """
        Load image from uploaded file.
        
        Args:
            file: Uploaded file
            
        Returns:
            PIL Image
        """
        contents = await file.read()
        return Image.open(BytesIO(contents)).convert("RGB")
        
    def _get_dummy_image(self) -> Image.Image:
        """
        Create dummy image for text-only processing.
        
        Returns:
            Black 224x224 PIL Image
        """
        return Image.new("RGB", self.config.image_size, color=(0, 0, 0))
        
    def _get_health_status(self) -> HealthResponse:
        """
        Get server health status.
        
        Returns:
            HealthResponse with status information
        """
        # Check model status
        models_loaded = {
            "openvla": hasattr(self, 'openvla_model') and self.openvla_model is not None,
            "clip": hasattr(self, 'clip_model') and self.clip_model is not None
        }
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.config.device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(self.config.device) / 1e9
        else:
            memory_allocated = 0
            memory_reserved = 0
            
        memory_usage = {
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved
        }
        
        return HealthResponse(
            status="healthy" if all(models_loaded.values()) else "degraded",
            models_loaded=models_loaded,
            device=self.config.device,
            memory_usage=memory_usage
        )
        
    def run(self):
        """Run the FastAPI server"""
        self.logger.info(f"Starting embedding server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level=self.config.log_level.lower()
        )

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the RT-Cache embedding server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9020,
        help="Server port"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create configuration
    config = ServerConfig()
    config.host = args.host
    config.port = args.port
    config.device = args.device
    config.workers = args.workers
    config.log_level = args.log_level
    
    # Create and run server
    server = EmbeddingServer(config)
    server.run()

if __name__ == "__main__":
    main()