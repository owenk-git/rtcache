#!/usr/bin/env python3
"""
VINN Evaluation Script
Evaluates trained VINN models using k-nearest neighbor retrieval
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import time
import os

# Import VINN models and dataset
from imitation_models.BC import TranslationModel, RotationModel, GripperModel
from train_BC_local import OpenXDataset, DATASETS
from local_embedding_fix import LocalBYOLEmbeddingExtractor

class VINNEvaluator:
    def __init__(self, model_dir, device='cuda', k_neighbors=5):
        self.device = device
        self.k_neighbors = k_neighbors
        self.model_dir = model_dir
        
        # Load trained models
        self.load_models()
        
        # Initialize embedding extractor
        self.embedding_extractor = LocalBYOLEmbeddingExtractor(device=device)
        
        print(f"✅ VINN Evaluator initialized with k={k_neighbors}")
    
    def load_models(self):
        """Load trained VINN models"""
        embedding_dim = 2048  # BYOL embedding dimension
        
        self.translation_model = TranslationModel(embedding_dim).to(self.device)
        self.rotation_model = RotationModel(embedding_dim).to(self.device)
        self.gripper_model = GripperModel(embedding_dim).to(self.device)
        
        # Load model weights
        trans_path = os.path.join(self.model_dir, 'translation_model.pth')
        rot_path = os.path.join(self.model_dir, 'rotation_model.pth')
        grip_path = os.path.join(self.model_dir, 'gripper_model.pth')
        
        if os.path.exists(trans_path):
            self.translation_model.load_state_dict(torch.load(trans_path, map_location=self.device))
            self.rotation_model.load_state_dict(torch.load(rot_path, map_location=self.device))
            self.gripper_model.load_state_dict(torch.load(grip_path, map_location=self.device))
            print("✅ Loaded trained VINN models")
        else:
            print("⚠️ No trained models found, using randomly initialized models")
        
        # Set to evaluation mode
        self.translation_model.eval()
        self.rotation_model.eval()
        self.gripper_model.eval()
    
    def build_database(self, database_size=5000):
        """Build embedding database for k-NN retrieval"""
        print(f"Building VINN database with {database_size} samples...")
        
        dataset = OpenXDataset(
            datasets=DATASETS[:10],  # Use subset for faster database building
            max_samples_per_dataset=database_size // 10,
            device=self.device
        )
        
        embeddings = []
        actions = []
        
        for i in range(min(len(dataset), database_size)):
            embedding, action = dataset[i]
            embeddings.append(embedding.cpu().numpy())
            actions.append(action.cpu().numpy())
        
        self.database_embeddings = np.array(embeddings)
        self.database_actions = np.array(actions)
        
        # Build k-NN index
        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors, metric='cosine')
        self.knn.fit(self.database_embeddings)
        
        print(f"✅ Database built: {len(self.database_embeddings)} samples")
    
    def predict_action_knn(self, embedding):
        """Predict action using k-NN retrieval (original VINN method)"""
        embedding = embedding.cpu().numpy().reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = self.knn.kneighbors(embedding)
        
        # Get actions from nearest neighbors
        neighbor_actions = self.database_actions[indices[0]]
        
        # Average the actions (simple aggregation)
        predicted_action = np.mean(neighbor_actions, axis=0)
        
        return predicted_action
    
    def predict_action_bc(self, embedding):
        """Predict action using trained BC models (current implementation)"""
        embedding = embedding.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            translation = self.translation_model(embedding).cpu().numpy()[0]
            rotation = self.rotation_model(embedding).cpu().numpy()[0]
            gripper_logits = self.gripper_model(embedding).cpu().numpy()[0]
            gripper = np.argmax(gripper_logits)  # Convert to class
        
        return np.concatenate([translation, rotation, [gripper]])
    
    def evaluate(self, test_samples=1000, method='bc'):
        """Evaluate VINN performance"""
        print(f"\n=== VINN Evaluation ({method.upper()}) ===")
        
        if method == 'knn':
            self.build_database()
        
        # Load test dataset
        test_dataset = OpenXDataset(
            datasets=DATASETS[-3:],  # Use different datasets for testing
            max_samples_per_dataset=test_samples // 3,
            device=self.device
        )
        
        predictions = []
        ground_truth = []
        inference_times = []
        
        print(f"Evaluating on {min(len(test_dataset), test_samples)} test samples...")
        
        for i in range(min(len(test_dataset), test_samples)):
            embedding, true_action = test_dataset[i]
            
            # Time the inference
            start_time = time.time()
            
            if method == 'knn':
                predicted_action = self.predict_action_knn(embedding)
            else:  # bc method
                predicted_action = self.predict_action_bc(embedding)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            predictions.append(predicted_action)
            ground_truth.append(true_action.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Separate continuous and discrete components
        trans_mse = mean_squared_error(ground_truth[:, :3], predictions[:, :3])
        rot_mse = mean_squared_error(ground_truth[:, 3:6], predictions[:, 3:6])
        
        # Gripper accuracy
        grip_true = ground_truth[:, 6].astype(int)
        grip_pred = predictions[:, 6].astype(int)
        grip_accuracy = np.mean(grip_true == grip_pred)
        
        # Overall MSE
        overall_mse = mean_squared_error(ground_truth, predictions)
        
        # Print results
        print(f"\n📊 VINN Evaluation Results:")
        print(f"Translation MSE: {trans_mse:.6f}")
        print(f"Rotation MSE: {rot_mse:.6f}")
        print(f"Gripper Accuracy: {grip_accuracy:.4f}")
        print(f"Overall MSE: {overall_mse:.6f}")
        print(f"Average Inference Time: {np.mean(inference_times)*1000:.2f}ms")
        
        return {
            'translation_mse': trans_mse,
            'rotation_mse': rot_mse,
            'gripper_accuracy': grip_accuracy,
            'overall_mse': overall_mse,
            'inference_time': np.mean(inference_times)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./vinn_local_models')
    parser.add_argument('--k_neighbors', type=int, default=5)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--method', type=str, default='bc', choices=['bc', 'knn'])
    parser.add_argument('--gpu', type=int, default=1)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Initialize evaluator
    evaluator = VINNEvaluator(
        model_dir=args.model_dir,
        device=device,
        k_neighbors=args.k_neighbors
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        test_samples=args.test_samples,
        method=args.method
    )
    
    # Save results
    import json
    results_path = f"{args.model_dir}/evaluation_results_{args.method}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")

if __name__ == '__main__':
    main()