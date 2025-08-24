#!/usr/bin/env python3
"""
BehaviorRetrieval Evaluation Script
Evaluates trained BehaviorRetrieval models with retrieval + BC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import time
import os
import pickle

# Import BehaviorRetrieval models and dataset
from train_local import VisualEncoder, VAE, BehaviorCloning, OpenXDataset, DATASETS

class BehaviorRetrievalEvaluator:
    def __init__(self, vae_model_path, bc_model_path, device='cuda'):
        self.device = device
        self.vae_model_path = vae_model_path
        self.bc_model_path = bc_model_path
        
        # Load trained models
        self.load_models()
        
        print(f"✅ BehaviorRetrieval Evaluator initialized")
    
    def load_models(self):
        """Load trained VAE and BC models"""
        # Initialize models
        self.vae = VAE(visual_dim=128, action_dim=7, latent_dim=128).to(self.device)
        self.bc = BehaviorCloning(latent_dim=128 + 7, action_dim=7).to(self.device)
        
        # Load model weights
        if os.path.exists(self.vae_model_path):
            self.vae.load_state_dict(torch.load(self.vae_model_path, map_location=self.device))
            print("✅ Loaded trained VAE model")
        else:
            print("⚠️ VAE model not found, using randomly initialized")
        
        if os.path.exists(self.bc_model_path):
            self.bc.load_state_dict(torch.load(self.bc_model_path, map_location=self.device))
            print("✅ Loaded trained BC model")
        else:
            print("⚠️ BC model not found, using randomly initialized")
        
        # Set to evaluation mode
        self.vae.eval()
        self.bc.eval()
    
    def build_retrieval_database(self, database_size=5000):
        """Build embedding database for retrieval"""
        print(f"Building retrieval database with {database_size} samples...")
        
        dataset = OpenXDataset(
            datasets=DATASETS[:10],  # Use subset for faster database building
            max_samples_per_dataset=database_size // 10,
            device=self.device
        )
        
        embeddings = []
        visual_features = []
        actions = []
        
        with torch.no_grad():
            for i in range(min(len(dataset), database_size)):
                visual_feat, action = dataset[i]
                
                # Compute VAE embedding (latent + action)
                embedding = self.vae.compute_embeddings(
                    visual_feat.unsqueeze(0).to(self.device),
                    action.unsqueeze(0).to(self.device)
                ).cpu().numpy()[0]
                
                embeddings.append(embedding)
                visual_features.append(visual_feat.cpu().numpy())
                actions.append(action.cpu().numpy())
        
        self.database_embeddings = np.array(embeddings)
        self.database_visual_features = np.array(visual_features)
        self.database_actions = np.array(actions)
        
        # Build k-NN index for retrieval
        self.knn = NearestNeighbors(n_neighbors=50, metric='cosine')  # Retrieve more for filtering
        self.knn.fit(self.database_embeddings)
        
        print(f"✅ Retrieval database built: {len(self.database_embeddings)} samples")
    
    def retrieve_demonstrations(self, query_visual_features, query_actions, retrieval_fraction=0.25):
        """Retrieve similar demonstrations for fine-tuning"""
        query_embedding = self.vae.compute_embeddings(
            query_visual_features.unsqueeze(0).to(self.device),
            query_actions.unsqueeze(0).to(self.device)
        ).cpu().numpy()
        
        # Find nearest neighbors
        n_retrieve = int(retrieval_fraction * len(self.database_embeddings))
        n_retrieve = max(n_retrieve, 50)  # At least 50 samples
        
        distances, indices = self.knn.kneighbors(query_embedding, n_neighbors=n_retrieve)
        
        # Return retrieved visual features and actions
        retrieved_visual = self.database_visual_features[indices[0]]
        retrieved_actions = self.database_actions[indices[0]]
        
        return retrieved_visual, retrieved_actions, distances[0]
    
    def finetune_bc(self, retrieved_visual, retrieved_actions, epochs=10, lr=0.0001):
        """Fine-tune BC model on retrieved demonstrations"""
        print(f"Fine-tuning BC on {len(retrieved_visual)} retrieved demonstrations...")
        
        # Convert to tensors
        retrieved_visual = torch.FloatTensor(retrieved_visual).to(self.device)
        retrieved_actions = torch.FloatTensor(retrieved_actions).to(self.device)
        
        # Get VAE embeddings for retrieved data
        with torch.no_grad():
            embeddings = self.vae.compute_embeddings(retrieved_visual, retrieved_actions)
        
        # Fine-tune BC
        optimizer = torch.optim.Adam(self.bc.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.bc.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predicted_actions = self.bc(embeddings)
            loss = criterion(predicted_actions, retrieved_actions)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"  Fine-tune epoch {epoch}: Loss={loss.item():.6f}")
        
        self.bc.eval()
        print(f"✅ BC fine-tuning completed")
    
    def predict_action(self, visual_features, action_context=None, use_retrieval=True, retrieval_fraction=0.25):
        """Predict action using BehaviorRetrieval pipeline"""
        visual_features = visual_features.to(self.device)
        
        if action_context is None:
            # Use zero action as context for first prediction
            action_context = torch.zeros(7).to(self.device)
        else:
            action_context = action_context.to(self.device)
        
        if use_retrieval:
            # Retrieve and fine-tune
            retrieved_visual, retrieved_actions, _ = self.retrieve_demonstrations(
                visual_features, action_context, retrieval_fraction
            )
            self.finetune_bc(retrieved_visual, retrieved_actions, epochs=5)
        
        # Predict using BC
        with torch.no_grad():
            embedding = self.vae.compute_embeddings(
                visual_features.unsqueeze(0),
                action_context.unsqueeze(0)
            )
            predicted_action = self.bc(embedding).cpu().numpy()[0]
        
        return predicted_action
    
    def evaluate(self, test_samples=1000, use_retrieval=True, retrieval_fraction=0.25):
        """Evaluate BehaviorRetrieval performance"""
        print(f"\n=== BehaviorRetrieval Evaluation ===")
        print(f"Retrieval: {'Enabled' if use_retrieval else 'Disabled'}")
        
        if use_retrieval:
            self.build_retrieval_database()
        
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
            visual_features, true_action = test_dataset[i]
            
            # Time the inference
            start_time = time.time()
            
            predicted_action = self.predict_action(
                visual_features,
                action_context=None,  # Could use previous action in sequence
                use_retrieval=use_retrieval,
                retrieval_fraction=retrieval_fraction
            )
            
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
        action_mse = mean_squared_error(ground_truth[:, :6], predictions[:, :6])
        
        # Gripper accuracy (if using classification)
        grip_mse = mean_squared_error(ground_truth[:, 6:], predictions[:, 6:])
        
        # Overall MSE
        overall_mse = mean_squared_error(ground_truth, predictions)
        
        # Print results
        print(f"\n📊 BehaviorRetrieval Evaluation Results:")
        print(f"Translation MSE: {trans_mse:.6f}")
        print(f"Rotation MSE: {rot_mse:.6f}")
        print(f"Action MSE (trans+rot): {action_mse:.6f}")
        print(f"Gripper MSE: {grip_mse:.6f}")
        print(f"Overall MSE: {overall_mse:.6f}")
        print(f"Average Inference Time: {np.mean(inference_times)*1000:.2f}ms")
        
        return {
            'translation_mse': trans_mse,
            'rotation_mse': rot_mse,
            'action_mse': action_mse,
            'gripper_mse': grip_mse,
            'overall_mse': overall_mse,
            'inference_time': np.mean(inference_times),
            'use_retrieval': use_retrieval
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_model', type=str, default='./br_local_models/vae.pth')
    parser.add_argument('--bc_model', type=str, default='./br_local_models/bc.pth')
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--use_retrieval', type=int, default=1, help='1 to use retrieval, 0 for direct BC')
    parser.add_argument('--retrieval_fraction', type=float, default=0.25)
    parser.add_argument('--gpu', type=int, default=1)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Initialize evaluator
    evaluator = BehaviorRetrievalEvaluator(
        vae_model_path=args.vae_model,
        bc_model_path=args.bc_model,
        device=device
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        test_samples=args.test_samples,
        use_retrieval=bool(args.use_retrieval),
        retrieval_fraction=args.retrieval_fraction
    )
    
    # Save results
    import json
    retrieval_suffix = "with_retrieval" if args.use_retrieval else "direct_bc"
    results_path = f"./br_local_models/evaluation_results_{retrieval_suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")

if __name__ == '__main__':
    main()