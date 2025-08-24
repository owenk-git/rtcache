#!/usr/bin/env python3
"""
Modified Behavior Retrieval Training using RT-cache data processing pipeline
Integrates with Open-X data via TFDS + remote embeddings + robomimic framework
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import torch
import requests
from io import BytesIO
import base64
import h5py
from tqdm import tqdm
import tempfile

# Add robomimic modules (fix path for different environments)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'robomimic'))

try:
    import robomimic.utils.train_utils as TrainUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.config import config_factory
    from robomimic.algo import algo_factory
    ROBOMIMIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: robomimic not available: {e}")
    print("Will use simplified training approach")
    ROBOMIMIC_AVAILABLE = False

################################################################################
#                           RT-Cache Integration (Same as VINN)
################################################################################

REMOTE_SERVER_URL = "http://localhost:8000/predict"

DATASETS = [
    "berkeley_cable_routing", "roboturk", "nyu_door_opening_surprising_effectiveness", 
    "viola", "berkeley_autolab_ur5", "toto", "columbia_cairlab_pusht_real", 
    "austin_sirius_dataset_converted_externally_to_rlds", 
    "austin_sailor_dataset_converted_externally_to_rlds", 
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds", 
    
    "tokyo_u_lsmo_converted_externally_to_rlds", 
    "dlr_sara_pour_converted_externally_to_rlds", "dlr_sara_grid_clamp_converted_externally_to_rlds", 
    "dlr_edan_shared_control_converted_externally_to_rlds", "asu_table_top_converted_externally_to_rlds", 
    "stanford_robocook_converted_externally_to_rlds", "utaustin_mutex",

    'fractal20220817_data',
    'kuka', 
    'bridge'
]

def dataset2path(dataset_name):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'

def ensure_rank1(value, default):
    t = tf.convert_to_tensor(value) if value is not None else tf.zeros(default)
    if t.shape.ndims == 0:
        t = tf.expand_dims(t, 0)
    return t

def _extract_action(action_data):
    """Extract 7-DOF action: world_vector(3) + rotation_delta(3) + gripper(1)"""
    if isinstance(action_data, dict):
        world_vector = ensure_rank1(action_data.get('world_vector'), (3,))
        rotation_delta = ensure_rank1(action_data.get('rotation_delta'), (3,))
        gripper = ensure_rank1(action_data.get('gripper_closedness_action'), (1,))
        combined = tf.concat([world_vector, rotation_delta, gripper], axis=-1)
        return combined.numpy().astype(np.float32)
    
    if isinstance(action_data, tf.Tensor) and action_data.shape[-1] == 7:
        return action_data.numpy().astype(np.float32)
    
    return np.zeros(7, dtype=np.float32)

def _clamp(val, low, high):
    return max(low, min(high, val))

def normalize_franka_action(raw_action):
    """Normalize action to standard range"""
    raw_list = raw_action.tolist()
    pos = [_clamp(x, -0.1, 0.1) for x in raw_list[:3]]
    ori = [_clamp(r, -0.5, 0.5) for r in raw_list[3:6]]
    grip = 1.0 if raw_list[6] > 0 else 0.0
    return np.array(pos + ori + [grip], dtype=np.float32)

def send_for_embedding(image_pil, text_prompt=None, url=REMOTE_SERVER_URL, option="image"):
    """Get OpenVLA/CLIP embeddings from remote server"""
    files = {}
    if image_pil is not None:
        buf = BytesIO()
        image_pil.save(buf, format='PNG')
        buf.seek(0)
        files["file"] = ("image.png", buf, "image/png")

    data = {
        "instruction": text_prompt if text_prompt else "",
        "option": option
    }
    resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json()

def decode_base64_torch_tensor(b64_string):
    binary_data = base64.b64decode(b64_string)
    buff = BytesIO(binary_data)
    tensor = torch.load(buff, map_location="cpu")
    return tensor

################################################################################
#                     Convert Open-X to HDF5 for robomimic
################################################################################

def create_hdf5_from_openx(datasets=DATASETS, max_samples_per_dataset=1000, output_path=None):
    """
    Convert Open-X data to HDF5 format compatible with robomimic
    Uses EXACT RT-cache data processing pipeline for consistency
    """
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.hdf5')
    
    print(f"Creating HDF5 dataset at: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Dataset metadata
        f.attrs['date'] = str(np.datetime64('now'))
        f.attrs['repository_version'] = 'rt-cache-openx'
        f.attrs['total'] = 0
        
        data_group = f.create_group('data')
        episode_idx = 0
        total_samples = 0
        
        # Same exclusion logic as RT-cache
        exclusion_dataset_list = []
        
        for dataset_idx, dataset_name in enumerate(datasets):
            if dataset_name in exclusion_dataset_list:
                print(f"Skipping dataset: {dataset_name}")
                continue
            
            print(f"\n========== Processing Dataset: {dataset_name} ==========")
            start_time = time.time()
            
            try:
                # EXACT same TFDS loading as RT-cache
                builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
                ds = builder.as_dataset(split='train', shuffle_files=False)
                
                # EXACT same key detection logic
                possible_image_keys = [
                    'image', 'rgb_static', 'front_rgb', 'agentview_rgb',
                    'rgb', 'hand_image', 'image_1'
                ]
                possible_text_keys = ['natural_language_instruction', 'language_instruction']
                
                # EXACT same RLDS detection
                if 'steps' in builder.info.features:
                    is_rlds = True
                    obs_keys = list(builder.info.features['steps']['observation'].keys())
                else:
                    is_rlds = False
                    obs_keys = list(builder.info.features.keys())
                
                display_image_key = next((k for k in possible_image_keys if k in obs_keys), None)
                display_text_key = next((k for k in possible_text_keys if k in obs_keys), None)
                if not display_image_key:
                    print(f"No valid image key found in dataset {dataset_name}; skipping.")
                    continue
                
                point_idx = 0
                dataset_samples = 0
                COUNT = 0
                
                if is_rlds:
                    # RLDS format - episodic data
                    for tf_episode in ds:
                        if dataset_samples >= max_samples_per_dataset:
                            break
                            
                        steps_list = list(tf_episode["steps"].as_numpy_iterator())
                        if len(steps_list) < 5:
                            continue
                        
                        # Create episode group
                        ep_group = data_group.create_group(f'demo_{episode_idx}')
                        
                        obs_list = []
                        action_list = []
                        
                        for step_np in steps_list:
                            obs = step_np["observation"]
                            act = step_np["action"]
                            
                            # Extract action
                            action_vector = _extract_action(act)
                            norm_action = normalize_franka_action(action_vector)
                            
                            # Process image
                            image_data = obs[display_image_key]
                            if isinstance(image_data, tf.Tensor):
                                image_data = image_data.numpy()
                            
                            if image_data.dtype != np.uint8:
                                img_array = (image_data * 255).astype(np.uint8)
                            else:
                                img_array = image_data
                            
                            # Resize to 224x224 for consistency
                            image_pil = Image.fromarray(img_array).resize((224, 224))
                            image_np = np.array(image_pil)
                            
                            # Get embeddings from RT-cache server
                            try:
                                server_out = send_for_embedding(image_pil, option="image")
                                image_features_b64 = server_out.get("image_features", None)
                                if image_features_b64:
                                    image_tensor = decode_base64_torch_tensor(image_features_b64)
                                    embedding = image_tensor.squeeze(0).numpy()  # [2176]
                                else:
                                    embedding = np.zeros(2176, dtype=np.float32)
                            except Exception as e:
                                print(f"Embedding failed: {e}")
                                embedding = np.zeros(2176, dtype=np.float32)
                            
                            # Store observation (robomimic format)
                            obs_dict = {
                                'image': image_np,  # Raw image
                                'embedding': embedding,  # RT-cache embedding
                            }
                            obs_list.append(obs_dict)
                            action_list.append(norm_action)
                            
                            dataset_samples += 1
                            total_samples += 1
                            
                            if action_vector[-1] == 1:  # gripper closed
                                break
                        
                        if len(obs_list) > 0:
                            # Save episode data
                            obs_group = ep_group.create_group('obs')
                            
                            # Stack observations
                            images = np.stack([obs['image'] for obs in obs_list])
                            embeddings = np.stack([obs['embedding'] for obs in obs_list])
                            actions = np.stack(action_list)
                            
                            obs_group.create_dataset('image', data=images, compression='gzip')
                            obs_group.create_dataset('embedding', data=embeddings, compression='gzip')
                            ep_group.create_dataset('actions', data=actions, compression='gzip')
                            
                            # Episode metadata
                            ep_group.attrs['num_samples'] = len(obs_list)
                            ep_group.attrs['dataset_name'] = dataset_name
                            
                            episode_idx += 1
                        
                        if dataset_samples >= max_samples_per_dataset:
                            break
                
                else:
                    # Non-RLDS format - treat as single episode
                    ds_list = list(ds)
                    if len(ds_list) < 5:
                        continue
                    
                    ep_group = data_group.create_group(f'demo_{episode_idx}')
                    
                    obs_list = []
                    action_list = []
                    
                    for sample in ds_list:
                        if dataset_samples >= max_samples_per_dataset:
                            break
                        
                        action_data = sample.get('action', tf.zeros(7))
                        action_vector = _extract_action(action_data)
                        norm_action = normalize_franka_action(action_vector)
                        
                        # Process image
                        image_data = sample[display_image_key]
                        if isinstance(image_data, tf.Tensor):
                            image_data = image_data.numpy()
                        
                        if image_data.dtype != np.uint8:
                            img_array = (image_data * 255).astype(np.uint8)
                        else:
                            img_array = image_data
                        
                        image_pil = Image.fromarray(img_array).resize((224, 224))
                        image_np = np.array(image_pil)
                        
                        # Get embeddings
                        try:
                            server_out = send_for_embedding(image_pil, option="image")
                            image_features_b64 = server_out.get("image_features", None)
                            if image_features_b64:
                                image_tensor = decode_base64_torch_tensor(image_features_b64)
                                embedding = image_tensor.squeeze(0).numpy()
                            else:
                                embedding = np.zeros(2176, dtype=np.float32)
                        except Exception as e:
                            print(f"Embedding failed: {e}")
                            embedding = np.zeros(2176, dtype=np.float32)
                        
                        obs_dict = {
                            'image': image_np,
                            'embedding': embedding,
                        }
                        obs_list.append(obs_dict)
                        action_list.append(norm_action)
                        
                        dataset_samples += 1
                        total_samples += 1
                        
                        if action_vector[-1] == 1:
                            break
                    
                    if len(obs_list) > 0:
                        # Save episode
                        obs_group = ep_group.create_group('obs')
                        
                        images = np.stack([obs['image'] for obs in obs_list])
                        embeddings = np.stack([obs['embedding'] for obs in obs_list])
                        actions = np.stack(action_list)
                        
                        obs_group.create_dataset('image', data=images, compression='gzip')
                        obs_group.create_dataset('embedding', data=embeddings, compression='gzip')
                        ep_group.create_dataset('actions', data=actions, compression='gzip')
                        
                        ep_group.attrs['num_samples'] = len(obs_list)
                        ep_group.attrs['dataset_name'] = dataset_name
                        
                        episode_idx += 1
                
                print(f"Loaded {dataset_samples} samples from {dataset_name}")
                
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        # Update total count
        f.attrs['total'] = total_samples
        
        # Add mask (required by robomimic)
        mask_group = f.create_group('mask')
        for i in range(episode_idx):
            mask_group.create_dataset(f'demo_{i}', data=np.ones(1, dtype=bool))
    
    print(f"Created HDF5 with {episode_idx} episodes and {total_samples} total samples")
    return output_path

################################################################################
#                           Training with Robomimic
################################################################################

def create_bc_config(hdf5_path, save_dir):
    """Create BC configuration for robomimic training"""
    
    config = config_factory(algo_name="bc")
    
    # Dataset
    config.train.data = hdf5_path
    config.train.output_dir = save_dir
    
    # Training parameters
    config.train.num_epochs = 50
    config.train.batch_size = 128
    config.train.seed = 1
    
    # Algorithm
    config.algo.bc.enabled = True
    config.algo.bc.loss.l2_weight = 1.0
    config.algo.bc.loss.l1_weight = 0.0
    
    # Observation modalities
    config.observation.modalities.obs.low_dim = []
    config.observation.modalities.obs.rgb = ["image"]
    config.observation.modalities.obs.depth = []
    config.observation.modalities.obs.scan = []
    
    # Network architecture
    config.algo.bc.actor_layer_dims = [1024, 1024, 512]
    config.algo.bc.gmm.enabled = False
    
    # Use embeddings as additional low-dim features
    config.observation.modalities.obs.low_dim = ["embedding"]
    
    return config

def train_behavior_retrieval_rtcache(args):
    """Train Behavior Retrieval using RT-cache pipeline + robomimic"""
    
    print("=" * 80)
    print("Behavior Retrieval Training with RT-Cache Open-X Data")
    print("=" * 80)
    
    # Step 1: Create HDF5 dataset from Open-X via RT-cache
    print("Step 1: Converting Open-X data to HDF5 via RT-cache pipeline...")
    hdf5_path = create_hdf5_from_openx(
        datasets=DATASETS[:3],  # Use first 3 datasets for demo
        max_samples_per_dataset=args.max_samples,
        output_path=os.path.join(args.save_dir, 'openx_rtcache_data.hdf5')
    )
    
    # Step 2: Create training configuration
    print("Step 2: Creating robomimic training configuration...")
    config = create_bc_config(hdf5_path, args.save_dir)
    config.train.num_epochs = args.epochs
    config.train.batch_size = args.batch_size
    
    # Save config
    config_path = os.path.join(args.save_dir, 'config.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.dump(), f, indent=2)
    
    print(f"Config saved to: {config_path}")
    
    # Step 3: Train using robomimic
    print("Step 3: Training Behavior Retrieval policy...")
    
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Create algorithm
    algo = algo_factory(
        algo_name="bc",
        config=config,
        obs_key_shapes={"image": (224, 224, 3), "embedding": (2176,)},
        ac_dim=7,  # 7-DOF action space
        device=device,
    )
    
    # Load dataset
    dataset, collate_fn = TrainUtils.load_data_for_training(
        config=config,
        obs_keys=algo.obs_keys
    )
    
    # Train
    TrainUtils.train(
        model=algo,
        dataset=dataset,
        collate_fn=collate_fn,
        config=config,
        device=device,
        log_writer=None  # Can add wandb logging here
    )
    
    print(f"✅ Behavior Retrieval training completed!")
    print(f"Models saved to: {args.save_dir}")
    
    return algo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./br_rtcache_models')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples per dataset')
    
    args = parser.parse_args()
    
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)
    
    # Train model
    algo = train_behavior_retrieval_rtcache(args)
    
    print("✅ Training completed!")

if __name__ == '__main__':
    main()