#!/usr/bin/env python3
"""
PROPER VINN Evaluation Script - Matching Original Algorithm
Implements the exact weighted k-NN mechanism from the original VINN paper
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import time
import os
import json

# Import the local dataset and models
from train_BC_local import OpenXDataset, DATASETS, LocalBYOLExtractor

class VINNProperEvaluator:
    def __init__(self, device='cuda', k=16):
        self.device = device
        self.k = k  # Number of nearest neighbors
        
        # Initialize BYOL encoder for generating embeddings
        self.byol_extractor = LocalBYOLExtractor(device=device)
        print(f"✅ VINN Proper Evaluator initialized with k={k}")
    
    def build_database(self, database_size=5000):
        """Build embedding database for k-NN retrieval"""
        print(f"Building VINN database with {database_size} samples...")
        
        dataset = OpenXDataset(
            datasets=DATASETS[:15],  # Use most datasets for database
            max_samples_per_dataset=database_size // 15,
            device=self.device
        )
        
        embeddings = []
        actions = []
        
        print("Extracting BYOL embeddings for database...")
        for i in range(min(len(dataset), database_size)):
            if i % 500 == 0:
                print(f"Processed {i}/{min(len(dataset), database_size)} samples")
                
            image, action = dataset[i]
            
            # Extract BYOL embedding (2048-D)
            with torch.no_grad():
                embedding = self.byol_extractor.extract_embedding(image.unsqueeze(0))
                embeddings.append(embedding.cpu().numpy()[0])
                actions.append(action.cpu().numpy())
        
        self.database_embeddings = np.array(embeddings)
        self.database_actions = np.array(actions)
        
        # Build k-NN index
        self.knn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        self.knn.fit(self.database_embeddings)
        
        print(f"✅ VINN database built: {len(self.database_embeddings)} samples")
    
    def dist_metric(self, x, y):
        """Distance metric from original VINN pseudocode"""
        return torch.norm(x - y).item()
    
    def calculate_action(self, dist_list, k):
        """
        Calculate action using weighted k-NN from original VINN pseudocode
        
        Args:
            dist_list: List of (distance, action) tuples
            k: Number of neighbors to use
        
        Returns:
            Weighted average action
        """
        action = torch.zeros(7)  # 7-DOF action space
        top_k_weights = torch.zeros(k)
        
        # Extract distances and apply softmax weighting
        for i in range(k):
            top_k_weights[i] = dist_list[i][0]
        
        # Apply softmax to negative distances (closer = higher weight)
        top_k_weights = F.softmax(-1 * top_k_weights, dim=0)
        
        # Weighted sum of actions
        for i in range(k):
            action = torch.add(top_k_weights[i] * dist_list[i][1], action)
        
        return action.numpy()
    
    def calculate_nearest_neighbors(self, query_img, k):
        """
        Original VINN nearest neighbor calculation
        
        Args:
            query_img: Input image tensor
            k: Number of neighbors
            
        Returns:
            Predicted action using weighted k-NN
        """
        # Extract query embedding
        with torch.no_grad():
            query_embedding = self.byol_extractor.extract_embedding(query_img.unsqueeze(0))
            query_embedding = query_embedding.cpu()
        
        dist_list = []
        
        # Calculate distances to all database samples
        for dataset_index in range(len(self.database_embeddings)):
            dataset_embedding = torch.FloatTensor(self.database_embeddings[dataset_index])
            dataset_action = torch.FloatTensor(self.database_actions[dataset_index])
            
            # Calculate distance
            distance = self.dist_metric(query_embedding.squeeze(0), dataset_embedding)
            dist_list.append((distance, dataset_action))
        
        # Sort by distance (ascending)
        dist_list = sorted(dist_list, key=lambda tup: tup[0])
        
        # Calculate weighted action using top-k neighbors
        pred_action = self.calculate_action(dist_list, k)
        
        return pred_action
    
    def predict_action(self, image):
        """Predict action using proper VINN algorithm"""
        return self.calculate_nearest_neighbors(image, self.k)
    
    def evaluate(self, test_samples=1000):
        """Evaluate VINN using proper weighted k-NN"""
        print(f"\n=== PROPER VINN Evaluation ===")
        print(f"Using k={self.k} nearest neighbors with softmax weighting")
        
        # Build database
        self.build_database()
        
        # Load test dataset (different from database)
        test_dataset = OpenXDataset(
            datasets=DATASETS[-4:],  # Use last 4 datasets for testing
            max_samples_per_dataset=test_samples // 4,
            device=self.device
        )
        
        predictions = []
        ground_truth = []
        inference_times = []
        
        print(f"Evaluating on {min(len(test_dataset), test_samples)} test samples...")
        
        for i in range(min(len(test_dataset), test_samples)):
            if i % 100 == 0:
                print(f"Processed {i}/{min(len(test_dataset), test_samples)} test samples")
                
            image, true_action = test_dataset[i]
            
            # Time the inference
            start_time = time.time()
            predicted_action = self.predict_action(image)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            predictions.append(predicted_action)
            ground_truth.append(true_action.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Separate metrics
        trans_mse = mean_squared_error(ground_truth[:, :3], predictions[:, :3])
        rot_mse = mean_squared_error(ground_truth[:, 3:6], predictions[:, 3:6])
        action_mse = mean_squared_error(ground_truth[:, :6], predictions[:, :6])
        
        # Gripper handling (binary classification accuracy)
        grip_true = (ground_truth[:, 6] > 0.5).astype(float)
        grip_pred = (predictions[:, 6] > 0.5).astype(float)
        grip_accuracy = np.mean(grip_true == grip_pred)
        
        # Overall MSE
        overall_mse = mean_squared_error(ground_truth, predictions)
        
        # Print results
        print(f"\n📊 PROPER VINN Evaluation Results:")
        print(f"k-NN neighbors: {self.k}")
        print(f"Translation MSE: {trans_mse:.6f}")
        print(f"Rotation MSE: {rot_mse:.6f}")
        print(f"Action MSE (trans+rot): {action_mse:.6f}")
        print(f"Gripper Accuracy: {grip_accuracy:.4f}")
        print(f"Overall MSE: {overall_mse:.6f}")
        print(f"Average Inference Time: {np.mean(inference_times)*1000:.2f}ms")
        
        return {
            'translation_mse': trans_mse,
            'rotation_mse': rot_mse,
            'action_mse': action_mse,
            'gripper_accuracy': grip_accuracy,
            'overall_mse': overall_mse,
            'inference_time': np.mean(inference_times),
            'k_neighbors': self.k,
            'method': 'proper_weighted_knn'
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--k', type=int, default=16, help='Number of nearest neighbors')
    parser.add_argument('--gpu', type=int, default=1)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Initialize evaluator
    evaluator = VINNProperEvaluator(device=device, k=args.k)
    
    # Run evaluation
    results = evaluator.evaluate(test_samples=args.test_samples)
    
    # Save results
    os.makedirs('./vinn_local_models', exist_ok=True)
    results_path = f"./vinn_local_models/evaluation_results_proper_knn.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")

if __name__ == '__main__':
    main()