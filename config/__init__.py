"""
RT-Cache Configuration Module

This module provides easy access to the centralized configuration system.
"""

from .rt_cache_config import (
    get_config,
    reload_config,
    update_config,
    get_database_config,
    get_server_config,
    get_paths_config,
    get_model_config,
    RTCacheConfig
)

__all__ = [
    'get_config',
    'reload_config',
    'update_config',
    'get_database_config',
    'get_server_config',
    'get_paths_config',
    'get_model_config',
    'RTCacheConfig'
]