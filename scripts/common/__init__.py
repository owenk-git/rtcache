"""
Common utilities extracted from duplicated code across RT-Cache scripts.

This module contains simple, practical utilities that are reused across
multiple scripts, avoiding code duplication without over-engineering.
"""

from .database import get_database_connections, DatabaseConnections
from .image_utils import (
    center_crop_to_square,
    resize_for_model,
    preprocess_image,
    image_to_base64,
    base64_to_image,
    decode_tensor_from_base64
)
from .embedding_client import create_embedding_client, EmbeddingClient

__all__ = [
    # Database utilities
    'get_database_connections',
    'DatabaseConnections',
    
    # Image processing utilities  
    'center_crop_to_square',
    'resize_for_model', 
    'preprocess_image',
    'image_to_base64',
    'base64_to_image',
    'decode_tensor_from_base64',
    
    # Embedding client
    'create_embedding_client',
    'EmbeddingClient'
]