#!/usr/bin/env python3
"""
Action Vector Generators for RT-Cache Data Collection

This module provides different action generation strategies for robot data collection.
Users can implement their own action generators by inheriting from ActionGenerator.

Author: RT-Cache Team
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import yaml


class ActionGenerator(ABC):
    """Base class for action vector generators"""
    
    @abstractmethod
    def get_action_vector(self, step_idx: int, episode_idx: str) -> List[float]:
        """
        Generate action vector for given step and episode.
        
        Args:
            step_idx: Current step index (1-based)
            episode_idx: Episode identifier
            
        Returns:
            List of action values (e.g., [x, y, z, rx, ry, rz, gripper])
        """
        pass
    
    @abstractmethod
    def get_episode_options(self) -> Dict[str, List[int]]:
        """
        Get available episode grouping options.
        
        Returns:
            Dictionary mapping option names to episode lists
        """
        pass


class DefaultActionGenerator(ActionGenerator):
    """
    Default action generator with predefined action sequences.
    
    This is the original hardcoded implementation, suitable for
    basic manipulation tasks with 3 objects and 3 viewpoints.
    """
    
    def __init__(self):
        self.episode_map = {
            "wrist": list(range(1, 10)),
            "side": list(range(10, 19)),
            "front": list(range(19, 28)),
            "all_object_1sample_allview": [1, 4, 7, 10, 13, 16, 19, 22, 25],
            "all_object_2sample_allview": [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26],
            "one_object_allview": [1, 2, 3, 10, 11, 12, 19, 20, 21],
            "two_object_allview": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24],
            "all": list(range(1, 28)),
        }
    
    def get_episode_options(self) -> Dict[str, List[int]]:
        """Get predefined episode grouping options"""
        return self.episode_map.copy()
    
    def get_action_vector(self, step_idx: int, episode_idx: str) -> List[float]:
        """
        Generate action vector using predefined trajectory patterns.
        
        This implements a 5-phase manipulation sequence:
        1. Approach (steps 1-5): Move towards object
        2. Descend (steps 6-8): Lower to grasp height  
        3. Grasp (steps 9-12): Close gripper and lift
        4. Transport (steps 13-17): Move to target location
        5. Release (steps 18+): Open gripper
        """
        def _pick(phase1, phase2, phase3, phase4, phase5):
            if 1 <= step_idx <= 5:
                return phase1
            elif 6 <= step_idx <= 8:
                return phase2
            elif 9 <= step_idx <= 12:
                return phase3
            elif 13 <= step_idx <= 17:
                return phase4
            else:
                return phase5

        # Define movement patterns for each object/viewpoint combination
        # L = Left, R = Right, F = Front; 1,2,3 = different objects
        _L1 = _pick([0, 0.035, 0], [0, 0, -0.055], [0, -0.02, 0], [0, 0, -0.055], [0, 0, 0])
        _R1 = _pick([0, -0.035, 0], [0, 0, -0.055], [0, 0.02, 0], [0, 0, -0.055], [0, 0, 0])
        _F1 = _pick([0.01, 0, 0], [0, 0, -0.055], [0, 0.01, 0], [0, 0, -0.055], [0, 0, 0])

        _L2 = _pick([0, 0.035, 0], [0, 0, -0.045], [-0.01, 0, 0], [0, 0, -0.045], [0, 0, 0])
        _R2 = _pick([0, -0.035, 0], [0, 0, -0.045], [-0.01, 0, 0], [0, 0, -0.045], [0, 0, 0])
        _F2 = _pick([0.02, 0, 0], [0, 0, -0.045], [-0.01, 0, 0], [0, 0, -0.045], [0, 0, 0])

        _L3 = _pick([0, 0.035, 0], [0, 0, -0.055], [0, 0.01, 0], [0, 0, -0.055], [0, 0, 0])
        _R3 = _pick([0, -0.035, 0], [0, 0, -0.055], [0, -0.01, 0], [0, 0, -0.055], [0, 0, 0])
        _F3 = _pick([0.01, 0, 0], [0, 0, -0.055], [-0.01, 0, 0], [0, 0, -0.055], [0, 0, 0])

        # Organize patterns by family (viewpoint) and variant (object)
        families = [[_L1, _L2, _L3], [_R1, _R2, _R3], [_F1, _F2, _F3]]

        try:
            eid = int(episode_idx)
            if not 1 <= eid <= 27:
                return [0, 0, 0]  # Zero action for invalid episodes
        except ValueError:
            return [0, 0, 0]  # Zero action for non-numeric episodes

        # Map episode ID to family (viewpoint) and variant (object)
        family = (eid - 1) % 3  # 0=Left, 1=Right, 2=Front
        variant = ((eid - 1) // 3) % 3  # 0=Obj1, 1=Obj2, 2=Obj3
        
        return families[family][variant]


class ConfigurableActionGenerator(ActionGenerator):
    """
    Configurable action generator that loads patterns from YAML files.
    
    This allows users to define custom action sequences without modifying code.
    """
    
    def __init__(self, config_path: str = "./config/action_patterns.yaml"):
        self.config_path = config_path
        self.patterns = {}
        self.episode_map = {}
        self._load_config()
    
    def _load_config(self):
        """Load action patterns from configuration file"""
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.patterns = config.get('action_patterns', {})
                self.episode_map = config.get('episode_map', {})
        except Exception as e:
            print(f"Warning: Could not load action config from {self.config_path}: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create a default configuration file as an example"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        default_config = {
            'action_patterns': {
                'simple_pick': {
                    'phases': {
                        'approach': {'steps': [1, 5], 'action': [0.02, 0, 0]},
                        'descend': {'steps': [6, 8], 'action': [0, 0, -0.05]},
                        'grasp': {'steps': [9, 12], 'action': [0, 0, 0]},
                        'lift': {'steps': [13, 15], 'action': [0, 0, 0.05]},
                        'release': {'steps': [16, 20], 'action': [0, 0, 0]}
                    }
                }
            },
            'episode_map': {
                'all': list(range(1, 28)),
                'test': [1, 2, 3]
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    def get_episode_options(self) -> Dict[str, List[int]]:
        """Get episode options from config"""
        return self.episode_map.copy()
    
    def get_action_vector(self, step_idx: int, episode_idx: str) -> List[float]:
        """Generate action vector based on loaded configuration"""
        # This is a simplified implementation - extend as needed
        if 'simple_pick' in self.patterns:
            pattern = self.patterns['simple_pick']['phases']
            
            for phase_name, phase_info in pattern.items():
                start, end = phase_info['steps']
                if start <= step_idx <= end:
                    return phase_info['action']
        
        return [0, 0, 0]  # Default zero action


class RandomActionGenerator(ActionGenerator):
    """
    Random action generator for exploration and testing.
    
    Useful for generating diverse training data or testing robustness.
    """
    
    def __init__(self, 
                 action_bounds: List[List[float]] = None,
                 seed: Optional[int] = None):
        """
        Initialize random action generator.
        
        Args:
            action_bounds: List of [min, max] bounds for each action dimension
            seed: Random seed for reproducibility
        """
        if action_bounds is None:
            # Default bounds for [x, y, z, rx, ry, rz, gripper]
            action_bounds = [[-0.05, 0.05]] * 3  # Position bounds
        
        self.action_bounds = action_bounds
        if seed is not None:
            np.random.seed(seed)
    
    def get_episode_options(self) -> Dict[str, List[int]]:
        """Get episode options for random generation"""
        return {
            "random": list(range(1, 100)),  # Support up to 100 episodes
            "test": list(range(1, 10))
        }
    
    def get_action_vector(self, step_idx: int, episode_idx: str) -> List[float]:
        """Generate random action vector within bounds"""
        action = []
        for min_val, max_val in self.action_bounds:
            action.append(np.random.uniform(min_val, max_val))
        return action


def create_action_generator(generator_type: str = "default", **kwargs) -> ActionGenerator:
    """
    Factory function to create action generators.
    
    Args:
        generator_type: Type of generator ("default", "configurable", "random")
        **kwargs: Additional arguments for specific generators
        
    Returns:
        ActionGenerator instance
    """
    if generator_type == "default":
        return DefaultActionGenerator()
    elif generator_type == "configurable":
        return ConfigurableActionGenerator(kwargs.get('config_path', './config/action_patterns.yaml'))
    elif generator_type == "random":
        return RandomActionGenerator(
            action_bounds=kwargs.get('action_bounds'),
            seed=kwargs.get('seed')
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


# Example usage
if __name__ == "__main__":
    # Test different generators
    generators = [
        ("Default", create_action_generator("default")),
        ("Random", create_action_generator("random", seed=42)),
    ]
    
    for name, generator in generators:
        print(f"\n{name} Generator:")
        print(f"Episode options: {list(generator.get_episode_options().keys())}")
        
        # Test a few actions
        for step in [1, 5, 10, 15]:
            action = generator.get_action_vector(step, "1")
            print(f"Step {step}: {action}")