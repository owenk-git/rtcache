"""
Results saving and logging utilities for retrieval system.

Extracted from retrieval_server.py to separate results management from core logic.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from PIL import Image


class ResultsManager:
    """Manages saving retrieval results and images."""
    
    def __init__(self, 
                 results_dir: str,
                 save_results: bool = True,
                 episode: Optional[int] = None):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory to save results
            save_results: Whether to save results
            episode: Episode number (auto-increment if None)
        """
        self.results_dir = Path(results_dir)
        self.save_results = save_results
        self.current_episode = episode
        self.step_counter = 0
        self.logger = logging.getLogger(__name__)
        
        if self.save_results:
            self.results_dir.mkdir(exist_ok=True)
    
    def get_episode_and_step(self) -> tuple[int, int]:
        """Get current episode and step numbers."""
        if self.current_episode is None:
            # Auto-increment episode based on step counter (17 steps per episode)
            episode = (self.step_counter // 17) + 1
            step = (self.step_counter % 17) + 1
        else:
            episode = self.current_episode
            step = self.step_counter + 1
        
        self.step_counter += 1
        return episode, step
    
    def create_method_directories(self, methods: List[str], embedding_type: str = "clip"):
        """Create directories for each method."""
        if not self.save_results:
            return
            
        for method in methods:
            if method == 'rtcache':
                method_name = f"rtcache-{embedding_type}"
            else:
                method_name = method
                
            method_dir = self.results_dir / method_name
            method_dir.mkdir(exist_ok=True)
    
    def save_current_image(self, 
                          image: Image.Image, 
                          episode: int, 
                          step: int,
                          methods: List[str],
                          embedding_type: str = "clip") -> List[str]:
        """
        Save current input image for each enabled method.
        
        Args:
            image: Current input image
            episode: Episode number
            step: Step number
            methods: List of enabled methods
            embedding_type: Embedding type for rtcache naming
            
        Returns:
            List of saved image paths
        """
        if not self.save_results:
            return []
        
        saved_paths = []
        filename = f"{step}_current_ep{episode}_step{step}.jpg"
        
        for method in methods:
            method_name = f"rtcache-{embedding_type}" if method == 'rtcache' else method
            method_dir = self.results_dir / method_name / str(episode)
            method_dir.mkdir(exist_ok=True)
            
            image_path = method_dir / filename
            image.save(image_path)
            saved_paths.append(str(image_path))
            self.logger.debug(f"Saved current image: {image_path}")
        
        return saved_paths
    
    def save_action_vectors(self, 
                           episode: int, 
                           step: int,
                           trajectories: Dict[str, Any],
                           methods: List[str],
                           embedding_type: str = "clip"):
        """
        Save action vectors as JSON for each method.
        
        Args:
            episode: Episode number
            step: Step number
            trajectories: Dictionary containing trajectory data for each method
            methods: List of enabled methods
            embedding_type: Embedding type for rtcache naming
        """
        if not self.save_results:
            return
        
        for method in methods:
            method_name = f"rtcache-{embedding_type}" if method == 'rtcache' else method
            method_dir = self.results_dir / method_name / str(episode)
            method_dir.mkdir(exist_ok=True)
            
            # Create action data structure
            action_data = {
                "episode": episode,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "action_vector": None
            }
            
            # Add method-specific trajectory data
            if method == 'rtcache' and method in trajectories:
                action_data["action_vector"] = trajectories[method][0] if trajectories[method] else None
                action_data["filtered_ids"] = trajectories.get("filtered_ids", [])
                action_data["all_trajectory"] = trajectories[method]
            elif f"{method}_trajectory" in trajectories:
                traj = trajectories[f"{method}_trajectory"]
                action_data["action_vector"] = traj[0] if traj else None
                action_data["trajectory"] = traj
            
            # Save action vector JSON
            action_file = method_dir / f"{step}_action_vector_ep{episode}_step{step}.json"
            with open(action_file, 'w') as f:
                json.dump(action_data, f, indent=2)
            
            self.logger.debug(f"Saved action vector: {action_file}")
    
    def save_retrieved_images(self, 
                             episode: int, 
                             step: int,
                             filtered_ids: List[str],
                             raw_images_dir: str,
                             embedding_type: str = "clip",
                             consecutive_steps: int = 2):
        """
        Save retrieved images for RT-Cache method.
        
        Args:
            episode: Episode number
            step: Step number
            filtered_ids: List of filtered sample IDs
            raw_images_dir: Directory containing raw images
            embedding_type: Embedding type for directory naming
            consecutive_steps: Number of consecutive steps to save
        """
        if not self.save_results or not filtered_ids:
            return
        
        rtcache_dir = self.results_dir / f"rtcache-{embedding_type}" / str(episode)
        rtcache_dir.mkdir(exist_ok=True)
        
        retrieved_count = 0
        retrieval_metadata = {
            "episode": episode,
            "step": step,
            "retrieved_samples": [],
            "total_retrieved_images": 0
        }
        
        for sample_id in filtered_ids:
            # Parse sample_id to get episode and step info
            parts = sample_id.split("_")
            if len(parts) >= 3:
                retrieved_ep = parts[1]
                retrieved_step = parts[2]
                
                # Save all consecutive steps for this sample
                for step_offset in range(consecutive_steps + 1):
                    actual_step = int(retrieved_step) + step_offset
                    raw_file = self._find_raw_image(raw_images_dir, retrieved_ep, actual_step)
                    
                    if raw_file and os.path.exists(raw_file):
                        retrieved_count += 1
                        retrieved_filename = f"{step}_retrieved_ep{retrieved_ep}_step{actual_step}.jpg"
                        dest_path = rtcache_dir / retrieved_filename
                        shutil.copy2(raw_file, dest_path)
                        self.logger.debug(f"Saved retrieved image: {dest_path}")
                
                # Add to metadata
                retrieval_metadata["retrieved_samples"].append({
                    "sample_id": sample_id,
                    "source_episode": retrieved_ep,
                    "source_step": retrieved_step,
                    "consecutive_steps": consecutive_steps + 1
                })
        
        retrieval_metadata["total_retrieved_images"] = retrieved_count
        
        # Save retrieval metadata
        if retrieved_count > 0:
            metadata_file = rtcache_dir / f"{step}_retrieval_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(retrieval_metadata, f, indent=2)
            self.logger.info(f"Saved {retrieved_count} retrieved images from {len(filtered_ids)} samples")
    
    def _find_raw_image(self, raw_dir: str, episode: str, step: int) -> Optional[str]:
        """
        Find raw image file with various naming conventions.
        
        Args:
            raw_dir: Base directory for raw images
            episode: Episode string
            step: Step number
            
        Returns:
            Path to image file if found, None otherwise
        """
        episode_dir = os.path.join(raw_dir, str(episode))
        if not os.path.exists(episode_dir):
            return None
        
        # Try various naming conventions
        candidates = [
            os.path.join(episode_dir, f"{step}.jpg"),
            os.path.join(episode_dir, f"{step}.png"),
            os.path.join(episode_dir, f"{step:02d}.jpg"),
            os.path.join(episode_dir, f"{step:02d}.png"),
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        return None
    
    def save_experiment_summary(self, summary_data: Dict[str, Any]):
        """Save experiment summary to JSON file."""
        if not self.save_results:
            return
            
        summary_file = self.results_dir / "experiment_summary.json"
        summary_data["saved_at"] = datetime.now().isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Saved experiment summary: {summary_file}")


def create_results_manager(config, save_results: bool = True, episode: Optional[int] = None) -> ResultsManager:
    """Factory function to create results manager."""
    return ResultsManager(
        results_dir=config.paths.results_dir,
        save_results=save_results,
        episode=episode
    )