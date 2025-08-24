"""
Common image processing utilities.

Extracted from duplicated functions across retrieval and collection servers.
"""

import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np


def center_crop_to_square(image: Image.Image) -> Image.Image:
    """
    Center crop image to square aspect ratio.
    
    Extracted from retrieval_server.py and data_collection_server.py.
    """
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_for_model(image: Image.Image, size: tuple = (224, 224)) -> Image.Image:
    """Resize image for model input."""
    return image.resize(size, Image.Resampling.LANCZOS)


def preprocess_image(image: Image.Image, size: tuple = (224, 224)) -> Image.Image:
    """
    Standard image preprocessing pipeline.
    
    1. Convert to RGB
    2. Center crop to square  
    3. Resize to target size
    """
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Center crop and resize
    image = center_crop_to_square(image)
    image = resize_for_model(image, size)
    
    return image


def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))


def decode_tensor_from_base64(b64_string: str) -> torch.Tensor:
    """
    Decode base64 encoded tensor.
    
    Used for embedding responses from servers.
    """
    bin_data = base64.b64decode(b64_string)
    buffer = BytesIO(bin_data)
    tensor = torch.load(buffer, map_location="cpu")
    return tensor