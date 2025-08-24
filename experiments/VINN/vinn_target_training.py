#!/usr/bin/env python3
"""
VINN Training with Your 27-Episode Target Dataset
Modified to use your specific data structure with get_action_vector mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import time
from tqdm import tqdm
from pathlib import Path

# Import VINN classes
from vinn_with_byol_training import ProperVINN, ImageOnlyDataset, DemonstrationDataset

################################################################################
#                    Target Episode Dataset Loader
################################################################################

def get_action_vector(i: int, epi: str):
    """Your original Franka canned plan action vector generation"""
    def f(a,b,c,d,e): return \
        a if 1<=i<=5 else b if 6<=i<=8 else c if 9<=i<=12 else d if 13<=i<=17 else e
    _L1 = f([0, 0.035,0], [0,0,-0.055], [0,-0.02,0],  [0,0,-0.055], [0,0,0])
    _R1 = f([0,-0.035,0], [0,0,-0.055], [0, 0.02,0],  [0,0,-0.055], [0,0,0])
    _F1 = f([0.01,0,0],  [0,0,-0.055], [0,0.01,0],  [0,0,-0.055], [0,0,0])

    _L2 = f([0, 0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _R2 = f([0,-0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _F2 = f([0.02,0,0],  [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    
    _L3 = f([0, 0.035,0], [0,0,-0.055], [0, 0.01,0],  [0,0,-0.055], [0,0,0])
    _R3 = f([0,-0.035,0], [0,0,-0.055], [0, -0.01,0],  [0,0,-0.055], [0,0,0])
    _F3 = f([0.01,0,0],  [0,0,-0.055], [-0.01,0,0],  [0,0,-0.055], [0,0,0])

    families  = [[_L1,_L2,_L3], [_R1,_R2,_R3], [_F1,_F2,_F3]]

    try:
        eid = int(epi)
    except ValueError:
        return [0,0,0]
    if not 1<=eid<=28: return [0,0,0]
    fam  = (eid-1) % 3
    var  = ((eid-1)//3) % 3
    return families[fam][var]

def load_target_episodes(root_dir):
    """Load your 27-episode target dataset"""
    print(f"🎯 Loading 27-episode target dataset from {root_dir}")
    
    episodes_data = []
    successful_episodes = 0
    total_steps = 0
    
    for episode_id in range(1, 28):  # Episodes 1-27
        episode_path = Path(root_dir) / str(episode_id)
        if not episode_path.exists():
            continue
            
        episode_steps = 0
        
        for step in range(1, 18):  # Steps 1-17
            # Try different image file formats
            img_file = episode_path / f"{step:02d}.jpg"
            if not img_file.exists():
                img_file = episode_path / f"{step}.jpg"
            
            if not img_file.exists():
                continue
                
            try:
                image = Image.open(img_file).convert("RGB")
                action_3d = get_action_vector(step, str(episode_id))
                # Expand to 7D: [x, y, z, roll, pitch, yaw, gripper]
                action_7d = np.array(action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
                episodes_data.append((image, action_7d))
                episode_steps += 1
                total_steps += 1
                
            except Exception as e:
                print(f"❌ Error loading {img_file}: {e}")
                continue
        
        if episode_steps > 0:
            successful_episodes += 1
            print(f"✅ Episode {episode_id}: {episode_steps} steps loaded")
    
    print(f"\n📊 Dataset Summary: {successful_episodes}/27 episodes, {total_steps} total steps")
    return episodes_data

class TargetEpisodeImageDataset(Dataset):
    """Dataset for BYOL training (images only)"""
    def __init__(self, target_data, transform=None):
        self.images = [img for img, _ in target_data]
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

class TargetEpisodeDemoDataset(Dataset):
    """Dataset for demonstration database (images + actions)"""
    def __init__(self, target_data):
        self.data = target_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

################################################################################
#                    VINN Training with Target Episodes
################################################################################

def train_vinn_on_target_episodes(args):
    """Train VINN on your 27-episode target dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your 27-episode target dataset
    print("=" * 60)
    print("🎯 LOADING YOUR 27-EPISODE TARGET DATASET")
    print("=" * 60)
    
    target_data = load_target_episodes(args.target_dir)
    
    if len(target_data) == 0:
        print("❌ No target data loaded! Check the path.")
        return
    
    print(f"✅ Loaded {len(target_data)} target samples")
    
    # Initialize VINN
    vinn = ProperVINN(device=device, k=args.k)
    
    # Phase 1: BYOL Visual Representation Learning on Target Images
    print("\n" + "=" * 60)
    print("🔧 PHASE 1: BYOL Training on Target Images")
    print("=" * 60)
    
    # Create image-only dataset for BYOL training
    byol_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_dataset = TargetEpisodeImageDataset(target_data, transform=byol_transform)
    
    # Train visual representation
    encoder = vinn.phase1_train_visual_representation(image_dataset, epochs=args.byol_epochs)
    
    # Phase 2: Build Demonstration Database with Target Episodes
    print("\n" + "=" * 60)
    print("🔍 PHASE 2: Building Demonstration Database")
    print("=" * 60)
    
    demo_dataset = TargetEpisodeDemoDataset(target_data)
    vinn.phase2_build_demonstration_database(demo_dataset)
    
    # Test inference
    print("\n" + "=" * 60)
    print("🧪 TESTING k-NN INFERENCE")
    print("=" * 60)
    
    # Test with first image from dataset
    test_image, expected_action = target_data[0]
    start_time = time.time()
    predicted_action = vinn.predict_action(test_image)
    inference_time = time.time() - start_time
    
    print(f"✅ Test Results:")
    print(f"  Expected action: {expected_action}")
    print(f"  Predicted action: {predicted_action}")
    print(f"  Inference time: {inference_time*1000:.2f}ms")
    print(f"  Action difference: {np.linalg.norm(expected_action - predicted_action):.6f}")
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    vinn.save_model(args.save_dir)
    
    # Save target data info
    import json
    metadata = {
        'target_dir': args.target_dir,
        'total_samples': len(target_data),
        'byol_epochs': args.byol_epochs,
        'k_neighbors': args.k,
        'training_completed': True
    }
    
    with open(f"{args.save_dir}/target_training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ VINN training on target episodes completed!")
    print(f"📁 Models saved to: {args.save_dir}")
    print(f"📊 Database size: {len(vinn.database_embeddings)} demonstrations")
    
    return vinn

def main():
    parser = argparse.ArgumentParser(description='VINN Training on 27-Episode Target Dataset')
    
    # Dataset parameters
    parser.add_argument('--target_dir', type=str, 
                        default='./data/rt-cache/raw/',
                        help='Path to your 27-episode target dataset')
    
    # Training parameters
    parser.add_argument('--byol_epochs', type=int, default=100,
                        help='BYOL training epochs (default: 100)')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of nearest neighbors (default: 16)')
    
    # System parameters
    parser.add_argument('--save_dir', type=str, default='./vinn_target_models',
                        help='Directory to save models (default: ./vinn_target_models)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with reduced epochs')
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick_test:
        print("🧪 Running in QUICK TEST mode...")
        args.byol_epochs = 20
        args.save_dir = './vinn_target_test'
    
    # Validate target directory
    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory not found: {args.target_dir}")
        print("Please check the path to your 27-episode dataset")
        return
    
    print("🚀 VINN TRAINING ON 27-EPISODE TARGET DATASET")
    print("=" * 60)
    print(f"📁 Target directory: {args.target_dir}")
    print(f"🔧 BYOL epochs: {args.byol_epochs}")
    print(f"🔍 k-NN neighbors: {args.k}")
    print(f"💾 Save directory: {args.save_dir}")
    print("=" * 60)
    
    # Run training
    vinn = train_vinn_on_target_episodes(args)
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Use model from: {args.save_dir}")

if __name__ == '__main__':
    main()