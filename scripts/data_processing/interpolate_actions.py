#!/usr/bin/env python3
"""
Action Interpolation Script for RT-Cache System

This script interpolates robot action trajectories to a unified control frequency
across different datasets. This is essential for creating consistent training
data from datasets with varying control frequencies.

Based on the original 1.1_data-interpolation.py
Author: RT-Cache Team
Date: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
import yaml
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import get_config

# Note: These imports are not available in the current structure
# Users should implement these classes or use direct implementations

def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Load environment variables and config
load_dotenv()
config = get_config()

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InterpolationConfig:
    """Configuration for action interpolation"""
    
    # Database settings
    mongo_url: str = config.database.mongo_url
    mongo_db: str = config.database.mongo_db_name
    
    # Interpolation settings
    target_frequency: float = config.dataset.target_frequency  # Hz
    target_dt: float = 1.0 / target_frequency  # seconds
    
    # Processing settings
    batch_size: int = config.dataset.processing_batch_size
    interpolation_method: str = config.dataset.interpolation_method  # nearest, linear, cubic
    
    # Logging
    log_level: str = config.system.log_level

# Dataset-specific configurations
DATASET_CONFIG = {
    "fractal20220817_data": {
        "control_freq": 3,
        "action_type": "eef_position",
    },
    "kuka": {
        "control_freq": 10,
        "action_type": "eef_position",
    },
    "bridge": {
        "control_freq": 5,
        "action_type": "eef_position",
    },
    "berkeley_cable_routing": {
        "control_freq": 10,
        "action_type": "eef_velocity",
    },
    "roboturk": {
        "control_freq": 10,
        "action_type": "eef_position",
    },
    "nyu_door_opening_surprising_effectiveness": {
        "control_freq": 3,
        "action_type": "eef_position",
    },
    "viola": {
        "control_freq": 20,
        "action_type": "eef_position",
    },
    "berkeley_autolab_ur5": {
        "control_freq": 5,
        "action_type": "eef_position",
    },
    "toto": {
        "control_freq": 30,
        "action_type": "joint_position",
    },
    "columbia_cairlab_pusht_real": {
        "control_freq": 10,
        "action_type": "eef_position",
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "control_freq": 20,
        "action_type": "eef_velocity",
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "control_freq": 20,
        "action_type": "eef_velocity",
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "control_freq": 10,
        "action_type": "eef_position",
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "control_freq": 10,
        "action_type": "eef_velocity",
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "control_freq": 10,
        "action_type": "eef_velocity",
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "control_freq": 10,
        "action_type": "eef_velocity",
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "control_freq": 5,
        "action_type": "eef_position",
    },
    "asu_table_top_converted_externally_to_rlds": {
        "control_freq": 12.5,
        "action_type": "eef_position",
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "control_freq": 5,
        "action_type": "eef_position",
    },
    "utaustin_mutex": {
        "control_freq": 20,
        "action_type": "eef_position",
    },
}

# ============================================================================
# Action Interpolation
# ============================================================================

class ActionInterpolator:
    """
    Interpolates robot action trajectories to unified control frequency.
    
    This class handles:
    - Loading trajectory data from MongoDB
    - Resampling to target frequency
    - Different interpolation methods
    - Velocity integration for position control
    """
    
    def __init__(self, config: InterpolationConfig):
        """
        Initialize action interpolator.
        
        Args:
            config: Interpolation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize MongoDB client
        self.mongo_client = MongoClient(config.mongo_url)
        self.database = self.mongo_client[config.mongo_db]
        
        # Statistics
        self.stats = {
            "episodes_processed": 0,
            "episodes_skipped": 0,
            "total_original_steps": 0,
            "total_interpolated_steps": 0
        }
        
    def interpolate_dataset(self, dataset_name: str):
        """
        Interpolate actions for an entire dataset.
        
        Args:
            dataset_name: Name of the dataset to process
        """
        if dataset_name not in DATASET_CONFIG:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return
            
        dataset_config = DATASET_CONFIG[dataset_name]
        original_freq = dataset_config["control_freq"]
        action_type = dataset_config["action_type"]
        
        self.logger.info(
            f"Interpolating {dataset_name}: {original_freq}Hz -> {self.config.target_frequency}Hz"
        )
        
        # Get collection name
        collection_name = f"{dataset_name}_collection"
        
        # Get all episodes for this dataset
        episodes = self._get_episodes(collection_name, dataset_name)
        
        self.logger.info(f"Found {len(episodes)} episodes to process")
        
        # Process each episode
        for episode_info in tqdm(episodes, desc=f"Processing {dataset_name}"):
            self._interpolate_episode(
                collection_name=collection_name,
                dataset_name=dataset_name,
                episode_idx=episode_info["episode_idx"],
                original_freq=original_freq,
                action_type=action_type
            )
            
        self.logger.info(f"Completed interpolation for {dataset_name}")
        
    def _get_episodes(self, collection_name: str, dataset_name: str) -> List[Dict]:
        """
        Get all episodes for a dataset.
        
        Args:
            collection_name: MongoDB collection name
            dataset_name: Dataset name
            
        Returns:
            List of episode information
        """
        # Get distinct episode indices
        pipeline = [
            {"$match": {"dataset_name": dataset_name}},
            {"$group": {
                "_id": "$episode_idx",
                "episode_idx": {"$first": "$episode_idx"},
                "step_count": {"$sum": 1},
                "max_step": {"$max": "$step_idx"}
            }}
        ]
        
        collection = self.database[collection_name]
        episodes = list(collection.aggregate(pipeline))
        
        return sorted(episodes, key=lambda x: x["episode_idx"])
        
    def _interpolate_episode(
        self,
        collection_name: str,
        dataset_name: str,
        episode_idx: int,
        original_freq: float,
        action_type: str
    ):
        """
        Interpolate actions for a single episode.
        
        Args:
            collection_name: MongoDB collection name
            dataset_name: Dataset name
            episode_idx: Episode index
            original_freq: Original control frequency
            action_type: Type of actions (eef_position, eef_velocity, joint_position)
        """
        # Load episode data
        collection = self.database[collection_name]
        episode_query = {
            "dataset_name": dataset_name,
            "episode_idx": episode_idx
        }
        episode_data = list(collection.find(episode_query).sort("step_idx", 1))
        
        if len(episode_data) < 2:
            self.logger.warning(f"Episode {episode_idx} too short, skipping")
            self.stats["episodes_skipped"] += 1
            return
            
        # Convert to DataFrame for easier processing
        df_episode = pd.DataFrame(episode_data)
        
        # Original timestep
        original_dt = 1.0 / original_freq
        
        # Check if interpolation is needed
        if abs(original_freq - self.config.target_frequency) < 0.01:
            self.logger.debug(f"Episode {episode_idx} already at target frequency")
            return
            
        # Perform interpolation
        interpolated_df = self._resample_episode(
            df_episode=df_episode,
            original_freq=original_freq,
            action_type=action_type
        )
        
        if interpolated_df is None:
            self.stats["episodes_skipped"] += 1
            return
            
        # Update database with interpolated data
        self._update_episode_in_db(
            collection_name=collection_name,
            dataset_name=dataset_name,
            episode_idx=episode_idx,
            interpolated_df=interpolated_df
        )
        
        # Update statistics
        self.stats["episodes_processed"] += 1
        self.stats["total_original_steps"] += len(df_episode)
        self.stats["total_interpolated_steps"] += len(interpolated_df)
        
    def _resample_episode(
        self,
        df_episode: pd.DataFrame,
        original_freq: float,
        action_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Resample a single episode to target frequency.
        
        Args:
            df_episode: Episode data as DataFrame
            original_freq: Original control frequency
            action_type: Type of actions
            
        Returns:
            Resampled DataFrame or None if failed
        """
        # Create time indices
        original_dt = 1.0 / original_freq
        episode_duration = len(df_episode) * original_dt
        
        # Original time points
        original_times = np.arange(0, episode_duration, original_dt)[:len(df_episode)]
        
        # Target time points
        target_times = np.arange(0, episode_duration, self.config.target_dt)
        
        # Extract action sequences
        if 'normalized_action' in df_episode.columns:
            actions = np.array(df_episode['normalized_action'].tolist())
        elif 'norm_action' in df_episode.columns:
            actions = np.array(df_episode['norm_action'].tolist())
        else:
            self.logger.error("No action data found in episode")
            return None
            
        # Interpolate based on method
        try:
            if self.config.interpolation_method == "nearest":
                interpolated_actions = self._interpolate_nearest(
                    original_times, actions, target_times
                )
            elif self.config.interpolation_method == "linear":
                interpolated_actions = self._interpolate_linear(
                    original_times, actions, target_times
                )
            elif self.config.interpolation_method == "cubic":
                interpolated_actions = self._interpolate_cubic(
                    original_times, actions, target_times
                )
            else:
                raise ValueError(f"Unknown interpolation method: {self.config.interpolation_method}")
                
        except Exception as e:
            self.logger.error(f"Interpolation failed: {e}")
            return None
            
        # Handle velocity integration if needed
        if action_type == "eef_velocity":
            interpolated_actions = self._integrate_velocity_to_position(
                interpolated_actions, self.config.target_dt
            )
            
        # Create new DataFrame
        interpolated_df = pd.DataFrame({
            'dataset_name': [df_episode.iloc[0]['dataset_name']] * len(target_times),
            'episode_idx': [df_episode.iloc[0]['episode_idx']] * len(target_times),
            'step_idx': list(range(len(target_times))),
            'normalized_action': interpolated_actions.tolist(),
            'interpolated': [True] * len(target_times),
            'original_freq': [original_freq] * len(target_times),
            'target_freq': [self.config.target_frequency] * len(target_times)
        })
        
        # Copy other fields from first original step
        for col in ['text', 'total_steps_in_episode']:
            if col in df_episode.columns:
                interpolated_df[col] = [df_episode.iloc[0][col]] * len(target_times)
                
        return interpolated_df
        
    def _interpolate_nearest(
        self,
        original_times: np.ndarray,
        actions: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Nearest neighbor interpolation."""
        from scipy.interpolate import interp1d
        
        interpolated = np.zeros((len(target_times), actions.shape[1]))
        
        for dim in range(actions.shape[1]):
            # Create interpolator
            interp_func = interp1d(
                original_times, actions[:, dim],
                kind='nearest',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Interpolate
            interpolated[:, dim] = interp_func(target_times)
            
        return interpolated
        
    def _interpolate_linear(
        self,
        original_times: np.ndarray,
        actions: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation."""
        from scipy.interpolate import interp1d
        
        interpolated = np.zeros((len(target_times), actions.shape[1]))
        
        for dim in range(actions.shape[1]):
            # Handle gripper separately (should be nearest)
            if dim == 6:  # Gripper dimension
                interp_func = interp1d(
                    original_times, actions[:, dim],
                    kind='nearest',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            else:
                interp_func = interp1d(
                    original_times, actions[:, dim],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
            interpolated[:, dim] = interp_func(target_times)
            
        return interpolated
        
    def _interpolate_cubic(
        self,
        original_times: np.ndarray,
        actions: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Cubic spline interpolation."""
        from scipy.interpolate import interp1d
        
        interpolated = np.zeros((len(target_times), actions.shape[1]))
        
        for dim in range(actions.shape[1]):
            # Handle gripper separately (should be nearest)
            if dim == 6:  # Gripper dimension
                kind = 'nearest'
            else:
                kind = 'cubic' if len(original_times) >= 4 else 'linear'
                
            interp_func = interp1d(
                original_times, actions[:, dim],
                kind=kind,
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            interpolated[:, dim] = interp_func(target_times)
            
        return interpolated
        
    def _integrate_velocity_to_position(
        self,
        velocity_actions: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Integrate velocity commands to position commands.
        
        Args:
            velocity_actions: Velocity actions array
            dt: Time step
            
        Returns:
            Position actions array
        """
        position_actions = velocity_actions.copy()
        
        # Integrate velocity to get relative position changes
        # Position dimensions are [x, y, z] (first 3 dimensions)
        for dim in range(3):
            # Convert velocity to position delta
            position_actions[:, dim] = velocity_actions[:, dim] * dt
            
        # Rotation and gripper remain the same
        
        return position_actions
        
    def _update_episode_in_db(
        self,
        collection_name: str,
        dataset_name: str,
        episode_idx: int,
        interpolated_df: pd.DataFrame
    ):
        """
        Update episode in database with interpolated data.
        
        Args:
            collection_name: MongoDB collection name
            dataset_name: Dataset name
            episode_idx: Episode index
            interpolated_df: Interpolated episode data
        """
        collection = self.database[collection_name]
        
        # Delete original episode
        delete_query = {
            "dataset_name": dataset_name,
            "episode_idx": episode_idx
        }
        collection.delete_many(delete_query)
        
        # Insert interpolated data
        documents = []
        for _, row in interpolated_df.iterrows():
            doc = {
                "id": f"{dataset_name}_{episode_idx}_{row['step_idx']}",
                "dataset_name": row["dataset_name"],
                "episode_idx": row["episode_idx"],
                "step_idx": row["step_idx"],
                "normalized_action": row["normalized_action"],
                "interpolated": row["interpolated"],
                "original_freq": row["original_freq"],
                "target_freq": row["target_freq"]
            }
            
            # Add optional fields
            if "text" in row and pd.notna(row["text"]):
                doc["text"] = row["text"]
            if "total_steps_in_episode" in row:
                doc["total_steps_in_episode"] = len(interpolated_df)
                
            documents.append(doc)
            
        if documents:
            collection.insert_many(documents)
            
        self.logger.debug(
            f"Updated episode {episode_idx}: {len(documents)} interpolated steps"
        )
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get interpolation statistics."""
        return {
            **self.stats,
            "compression_ratio": (
                self.stats["total_original_steps"] / max(1, self.stats["total_interpolated_steps"])
            )
        }

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Interpolate robot action trajectories to unified frequency"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets to interpolate (or 'all')"
    )
    parser.add_argument(
        "--target-freq",
        type=float,
        default=10.0,
        help="Target control frequency in Hz"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="nearest",
        choices=["nearest", "linear", "cubic"],
        help="Interpolation method"
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
    logger = logging.getLogger(__name__)
    
    # Create configuration (override with command line args if provided)
    interp_config = InterpolationConfig()
    if hasattr(args, 'target_freq'):
        interp_config.target_frequency = args.target_freq
        interp_config.target_dt = 1.0 / args.target_freq
    if hasattr(args, 'method'):
        interp_config.interpolation_method = args.method
    if hasattr(args, 'log_level'):
        interp_config.log_level = args.log_level
    
    # Get dataset list
    if args.datasets.lower() == 'all':
        dataset_names = list(DATASET_CONFIG.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(',')]
        
    # Validate datasets
    invalid_datasets = [d for d in dataset_names if d not in DATASET_CONFIG]
    if invalid_datasets:
        logger.error(f"Unknown datasets: {invalid_datasets}")
        return
        
    logger.info(f"Interpolating {len(dataset_names)} datasets to {args.target_freq}Hz")
    
    # Create interpolator
    interpolator = ActionInterpolator(interp_config)
    
    # Process datasets
    for dataset_name in dataset_names:
        try:
            interpolator.interpolate_dataset(dataset_name)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            continue
            
    # Print final statistics
    stats = interpolator.get_statistics()
    logger.info("Interpolation Statistics:")
    logger.info(f"  Episodes processed: {stats['episodes_processed']}")
    logger.info(f"  Episodes skipped: {stats['episodes_skipped']}")
    logger.info(f"  Original steps: {stats['total_original_steps']}")
    logger.info(f"  Interpolated steps: {stats['total_interpolated_steps']}")
    logger.info(f"  Compression ratio: {stats['compression_ratio']:.2f}")
    
    logger.info("Action interpolation complete!")

if __name__ == "__main__":
    main()