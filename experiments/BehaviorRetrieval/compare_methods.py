#!/usr/bin/env python3
"""
Compare VINN vs BehaviorRetrieval Performance
"""

import json
import argparse
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

def run_vinn_evaluation(vinn_dir, samples=1000):
    """Run PROPER VINN evaluation with weighted k-NN"""
    print("🔄 Running PROPER VINN evaluation (weighted k-NN)...")
    
    cmd = [
        'python', f'{vinn_dir}/evaluate_vinn_proper.py',
        '--test_samples', str(samples),
        '--k', '16'  # Standard k value
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=vinn_dir)
    if result.returncode != 0:
        print(f"❌ VINN evaluation failed: {result.stderr}")
        return None
    
    # Load results
    results_path = f"{vinn_dir}/vinn_local_models/evaluation_results_proper_knn.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def run_br_evaluation(br_dir, samples=1000, use_retrieval=True):
    """Run BehaviorRetrieval evaluation"""
    method = "with retrieval" if use_retrieval else "direct BC"
    print(f"🔄 Running BehaviorRetrieval evaluation ({method})...")
    
    cmd = [
        'python', f'{br_dir}/evaluate_br.py',
        '--vae_model', f'{br_dir}/br_local_models/vae.pth',
        '--bc_model', f'{br_dir}/br_local_models/bc.pth',
        '--test_samples', str(samples),
        '--use_retrieval', str(int(use_retrieval))
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=br_dir)
    if result.returncode != 0:
        print(f"❌ BehaviorRetrieval evaluation failed: {result.stderr}")
        return None
    
    # Load results
    suffix = "with_retrieval" if use_retrieval else "direct_bc"
    results_path = f"{br_dir}/br_local_models/evaluation_results_{suffix}.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def compare_results(vinn_results, br_results, br_direct_results=None):
    """Compare and visualize results"""
    print("\n" + "="*60)
    print("📊 VINN vs BehaviorRetrieval Comparison")
    print("="*60)
    
    if vinn_results is None or br_results is None:
        print("❌ Missing evaluation results")
        return
    
    # Create comparison table
    print(f"{'Metric':<25} {'VINN':<15} {'BR (Retrieval)':<15} {'BR (Direct)':<15}")
    print("-" * 70)
    
    metrics = [
        ('Translation MSE', 'translation_mse'),
        ('Rotation MSE', 'rotation_mse'),
        ('Overall MSE', 'overall_mse'),
        ('Inference Time (ms)', 'inference_time')
    ]
    
    for metric_name, metric_key in metrics:
        vinn_val = vinn_results.get(metric_key, 0)
        br_val = br_results.get(metric_key, 0)
        
        if metric_key == 'inference_time':
            vinn_val *= 1000  # Convert to ms
            br_val *= 1000
        
        # Format values
        if metric_key == 'inference_time':
            vinn_str = f"{vinn_val:.2f}"
            br_str = f"{br_val:.2f}"
        else:
            vinn_str = f"{vinn_val:.6f}"
            br_str = f"{br_val:.6f}"
        
        # Add direct BR results if available
        br_direct_str = "N/A"
        if br_direct_results and metric_key in br_direct_results:
            br_direct_val = br_direct_results[metric_key]
            if metric_key == 'inference_time':
                br_direct_val *= 1000
                br_direct_str = f"{br_direct_val:.2f}"
            else:
                br_direct_str = f"{br_direct_val:.6f}"
        
        print(f"{metric_name:<25} {vinn_str:<15} {br_str:<15} {br_direct_str:<15}")
    
    # Special handling for gripper
    if 'gripper_accuracy' in vinn_results:
        print(f"{'Gripper Accuracy':<25} {vinn_results['gripper_accuracy']:.4f}      N/A             N/A")
    if 'gripper_mse' in br_results:
        print(f"{'Gripper MSE':<25} N/A             {br_results['gripper_mse']:.6f}      {br_direct_results.get('gripper_mse', 'N/A')}")
    
    # Analysis
    print("\n🔍 Analysis:")
    
    # Translation comparison
    if vinn_results['translation_mse'] < br_results['translation_mse']:
        print("✅ VINN has better translation accuracy")
    else:
        print("✅ BehaviorRetrieval has better translation accuracy")
    
    # Speed comparison  
    vinn_speed = vinn_results['inference_time'] * 1000
    br_speed = br_results['inference_time'] * 1000
    
    if vinn_speed < br_speed:
        print(f"⚡ VINN is faster ({vinn_speed:.2f}ms vs {br_speed:.2f}ms)")
    else:
        print(f"⚡ BehaviorRetrieval is faster ({br_speed:.2f}ms vs {vinn_speed:.2f}ms)")
    
    # Create visualization
    create_comparison_plot(vinn_results, br_results, br_direct_results)

def create_comparison_plot(vinn_results, br_results, br_direct_results=None):
    """Create comparison visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    methods = ['VINN', 'BR (Retrieval)']
    if br_direct_results:
        methods.append('BR (Direct)')
    
    # Translation MSE
    trans_values = [vinn_results['translation_mse'], br_results['translation_mse']]
    if br_direct_results:
        trans_values.append(br_direct_results['translation_mse'])
    
    ax1.bar(methods, trans_values, color=['blue', 'red', 'orange'][:len(methods)])
    ax1.set_title('Translation MSE (Lower is Better)')
    ax1.set_ylabel('MSE')
    
    # Rotation MSE
    rot_values = [vinn_results['rotation_mse'], br_results['rotation_mse']]
    if br_direct_results:
        rot_values.append(br_direct_results['rotation_mse'])
    
    ax2.bar(methods, rot_values, color=['blue', 'red', 'orange'][:len(methods)])
    ax2.set_title('Rotation MSE (Lower is Better)')
    ax2.set_ylabel('MSE')
    
    # Overall MSE
    overall_values = [vinn_results['overall_mse'], br_results['overall_mse']]
    if br_direct_results:
        overall_values.append(br_direct_results['overall_mse'])
    
    ax3.bar(methods, overall_values, color=['blue', 'red', 'orange'][:len(methods)])
    ax3.set_title('Overall MSE (Lower is Better)')
    ax3.set_ylabel('MSE')
    
    # Inference Time
    time_values = [vinn_results['inference_time']*1000, br_results['inference_time']*1000]
    if br_direct_results:
        time_values.append(br_direct_results['inference_time']*1000)
    
    ax4.bar(methods, time_values, color=['blue', 'red', 'orange'][:len(methods)])
    ax4.set_title('Inference Time (Lower is Better)')
    ax4.set_ylabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig('./method_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison plot saved as 'method_comparison.png'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vinn_dir', type=str, default='./VINN')
    parser.add_argument('--br_dir', type=str, default='./BehaviorRetrieval')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--skip_vinn', action='store_true', help='Skip VINN evaluation')
    parser.add_argument('--skip_br', action='store_true', help='Skip BR evaluation')
    
    args = parser.parse_args()
    
    vinn_results = None
    br_results = None
    br_direct_results = None
    
    # Run evaluations
    if not args.skip_vinn:
        vinn_results = run_vinn_evaluation(args.vinn_dir, args.samples)
    else:
        # Try to load existing results
        results_path = f"{args.vinn_dir}/vinn_local_models/evaluation_results_proper_knn.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                vinn_results = json.load(f)
    
    if not args.skip_br:
        br_results = run_br_evaluation(args.br_dir, args.samples, use_retrieval=True)
        br_direct_results = run_br_evaluation(args.br_dir, args.samples, use_retrieval=False)
    else:
        # Try to load existing results
        results_path1 = f"{args.br_dir}/br_local_models/evaluation_results_with_retrieval.json"
        results_path2 = f"{args.br_dir}/br_local_models/evaluation_results_direct_bc.json"
        
        if os.path.exists(results_path1):
            with open(results_path1, 'r') as f:
                br_results = json.load(f)
        
        if os.path.exists(results_path2):
            with open(results_path2, 'r') as f:
                br_direct_results = json.load(f)
    
    # Compare results
    compare_results(vinn_results, br_results, br_direct_results)

if __name__ == '__main__':
    main()