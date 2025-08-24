#!/usr/bin/env python3

import os
import time
import base64
import math
import shutil  # for copying files
from datetime import datetime
from io import BytesIO
import numpy as np
import torch
import requests

# Flask
from flask import Flask, request, jsonify, send_from_directory

from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult

from PIL import Image
from pymongo import MongoClient

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Additional imports for VINN and BehaviorRetrieval
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json
from pathlib import Path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import get_config

###############################################################################
# Command Line Arguments
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='RT-Cache Retrieval Server with VINN and BehaviorRetrieval')
    
    # Model selection
    parser.add_argument('--models', nargs='+', choices=['rtcache', 'vinn', 'br'], 
                        default=['rtcache', 'vinn', 'br'],
                        help='Models to load (default: all models)')
    
    # Embedding type
    parser.add_argument('--embedding', choices=['dino_siglip', 'clip'], default='clip',
                        help='Embedding type to use (default: clip)')
    
    # Server configuration
    parser.add_argument('--port', type=int, default=5002, help='Server port (default: 5002)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--embedding-url', type=str, default='http://127.0.0.1:9020/predict',
                        help='Embedding server URL (default: http://127.0.0.1:9020/predict)')
    
    # Database configuration
    parser.add_argument('--dataset', type=str, default='test', help='Dataset name (default: test)')
    parser.add_argument('--collection-suffix', type=str, default='clip', 
                        help='Collection name suffix (default: DINOSligLIP, )')
    
    # Model paths
    parser.add_argument('--vinn-model-dir', type=str, default='../VINN/vinn_target_models',
                        help='VINN model directory')
    parser.add_argument('--br-model-dir', type=str, default='../BehaviorRetrieval/br_target_models',
                        help='BehaviorRetrieval model directory')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    
    # Results saving configuration
    parser.add_argument('--save-results', action='store_true', 
                        help='Save results for visualization')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Results directory (default: ./results)')
    parser.add_argument('--episode', type=int, default=None,
                        help='Episode number for result saving (auto-increment if not specified)')
    
    return parser.parse_args()

# Parse arguments and get config
args = parse_args()
config = get_config()

###############################################################################
# Configuration
###############################################################################
RUN_SERVER           = True  
REMOTE_EMBEDDING_URL = args.embedding_url if hasattr(args, 'embedding_url') and args.embedding_url != 'http://127.0.0.1:9020/predict' else config.server.embedding_url

# Embedding selection based on args or config
EMBEDDING_TYPE = args.embedding if hasattr(args, 'embedding') else config.retrieval.embedding_type

# Dataset configuration
DATASET_NAME         = args.dataset if hasattr(args, 'dataset') and args.dataset != 'test' else config.dataset.active_datasets[0] if config.dataset.active_datasets else 'test'
MONGO_URL            = config.database.mongo_url
DB_NAME              = config.database.mongo_db_name
ID_FIELD             = config.retrieval.id_field
COLL_NAME            = DATASET_NAME + "_collection"

QDRANT_HOST          = config.database.qdrant_host
QDRANT_PORT          = config.database.qdrant_port
QDRANT_COLLECTION    = f"image_collection_{DATASET_NAME}_{args.collection_suffix if hasattr(args, 'collection_suffix') else 'clip'}"
CENTROID_COLL_NAME   = f"image_collection_{DATASET_NAME}_{args.collection_suffix if hasattr(args, 'collection_suffix') else 'clip'}"

DB_LIMIT_NUM         = config.retrieval.max_results
NUM_CANDIDATES       = config.retrieval.num_candidates        # how many doc-IDs we keep
CONSECUTIVE_STEPS    = config.retrieval.consecutive_steps        # how many consecutive steps
ZERO_ACTION          = [0.0]*7

# Raw images live here: final_images/raw/<episode>/<step>.jpg
RAW_BASE_DIR         = config.paths.image_storage_path

# Rotating logs after 17 calls
LOGS_BASE_DIR        = os.path.join(config.paths.image_storage_path, "logs")
EPISODE_LENGTH       = 17
call_count           = 0  # increments each pipeline call

# Model paths and device configuration
VINN_MODEL_DIR       = args.vinn_model_dir if hasattr(args, 'vinn_model_dir') else config.paths.model_cache_dir
BR_MODEL_DIR         = args.br_model_dir if hasattr(args, 'br_model_dir') else config.paths.model_cache_dir
if hasattr(args, 'device') and args.device == 'auto':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif hasattr(args, 'device'):
    DEVICE = args.device
else:
    DEVICE = config.model.device

# Model selection flags
models_to_load = args.models if hasattr(args, 'models') else ['rtcache']
LOAD_RTCACHE = 'rtcache' in models_to_load
LOAD_VINN = 'vinn' in models_to_load
LOAD_BR = 'br' in models_to_load

# Results saving configuration
SAVE_RESULTS = args.save_results if hasattr(args, 'save_results') else config.experiment.save_results
RESULTS_DIR = args.results_dir if hasattr(args, 'results_dir') else config.paths.results_dir
CURRENT_EPISODE = args.episode if hasattr(args, 'episode') else None
STEP_COUNTER = 0  # Auto-increment step counter

# Initialize results directories if saving is enabled
if SAVE_RESULTS:
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    models_to_load = args.models if hasattr(args, 'models') else ['rtcache']
    for method in ['rtcache', 'vinn', 'br']:
        if method in models_to_load:
            if method == 'rtcache':
                # Create specific RT-Cache directory based on embedding type
                method_name = f"rtcache-{EMBEDDING_TYPE}"
                method_dir = Path(RESULTS_DIR) / method_name
            else:
                method_dir = Path(RESULTS_DIR) / method
            method_dir.mkdir(exist_ok=True)

models_to_load = args.models if hasattr(args, 'models') else ['rtcache']
server_port = args.port if hasattr(args, 'port') else config.server.retrieval_port
print(f"🚀 Server Configuration:")
print(f"   Models to load: {models_to_load}")
print(f"   Embedding type: {EMBEDDING_TYPE}")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Collection: {QDRANT_COLLECTION}")
print(f"   Device: {DEVICE}")
print(f"   Port: {server_port}")
print(f"   Save results: {SAVE_RESULTS}")
if SAVE_RESULTS:
    print(f"   Results directory: {RESULTS_DIR}")
    print(f"   Episode: {CURRENT_EPISODE if CURRENT_EPISODE else 'auto-increment'}")
print()

###############################################################################
# Flask
###############################################################################
app = Flask(__name__)

###############################################################################
# 1) Qdrant Store that Skips text=None
###############################################################################
class MyQdrantStore(QdrantVectorStore):
    def parse_to_query_result(self, search_result):
        nodes = []
        similarities = []
        # Only keep docs from DATASET_NAME
        specific_datasets = [DATASET_NAME]

        for point in search_result:
            payload = point.payload or {}
            if payload.get("dataset_name") not in specific_datasets:
                continue

            # skip if text missing
            if payload.get("text") is None:
                continue

            node = TextNode(
                text=payload["text"],
                id_=str(point.id),
                metadata=payload
            )
            nodes.append(node)
            similarities.append(
                point.score if hasattr(point, "score") else 0.0
            )

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities)

###############################################################################
# 2) Connect to Mongo + Qdrant & Preload docs
###############################################################################
print(f"### Using embedding type: {EMBEDDING_TYPE}")
print("### Connecting to Mongo ...")
mongo_client = MongoClient(MONGO_URL)
db          = mongo_client[DB_NAME]
mongo_coll  = db[COLL_NAME]

print("### Pre-loading docs into memory ...")
doc_cache = {}
cursor = mongo_coll.find({}, {ID_FIELD:1, "raw_action":1, "_id":0})
count = 0
for doc in cursor:
    _id = doc.get(ID_FIELD, None)
    if _id:
        doc_cache[_id] = doc
        count += 1
print(f"### Loaded {count} documents into doc_cache.\n")

print("### Connecting to Qdrant ...")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)
image_store = MyQdrantStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    content_key="logical_id"
)
print("### Qdrant + LlamaIndex setup complete.\n")

###############################################################################
# 3) Rotating Log Folder
###############################################################################
def _new_log_dir():
    """
    final_images/logs/20250504_104454/   (for example)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(LOGS_BASE_DIR, ts)
    os.makedirs(folder, exist_ok=True)
    return folder

CURRENT_LOG_DIR = _new_log_dir()
print(f"[INFO] Initial logs folder => {CURRENT_LOG_DIR}")

###############################################################################
# 4) Embedding / Utility
###############################################################################
def decode_base64_torch_tensor(b64_string: str) -> torch.Tensor:
    bin_data = base64.b64decode(b64_string)
    buff = BytesIO(bin_data)
    tensor = torch.load(buff, map_location="cpu")
    return tensor

def center_crop_to_square(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return pil_image.crop((left, top, left + side, top + side))

def send_image_with_prompt(pil_image: Image.Image, prompt: str, url: str, option: str) -> dict:
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("image.jpg", buf, "image/jpeg")}
    data  = {"instruction": prompt, "option": option}
    resp  = requests.post(url, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()

def get_image_embedding(pil_img: Image.Image) -> torch.Tensor:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("image.jpg", buf, "image/jpeg")}
    data  = {"instruction": "", "option": "image"}
    resp  = requests.post(REMOTE_EMBEDDING_URL, files=files, data=data, timeout=300)
    resp.raise_for_status()
    resp_json = resp.json()
    
    # Select embedding type based on configuration
    if EMBEDDING_TYPE == "clip":
        if "clip_image_features" not in resp_json:
            raise ValueError("No 'clip_image_features' found in server response.")
        b64_str = resp_json["clip_image_features"]
    else:  # default to dino_siglip
        if "image_features" not in resp_json:
            raise ValueError("No 'image_features' found in server response.")
        b64_str = resp_json["image_features"]
    
    bin_data = base64.b64decode(b64_str)
    buff = BytesIO(bin_data)
    tensor = torch.load(buff, map_location="cpu")
    return tensor

###############################################################################
# 5) Results Saving Functions
###############################################################################

def get_method_directory_name(method):
    """Get the correct directory name for a method"""
    if method == 'rtcache':
        return f"rtcache-{EMBEDDING_TYPE}"
    else:
        return method

def get_episode_and_step():
    """Get current episode and step numbers"""
    global STEP_COUNTER, CURRENT_EPISODE
    
    if CURRENT_EPISODE is None:
        # Auto-increment episode based on step counter
        episode = (STEP_COUNTER // 17) + 1  # 17 steps per episode
        step = (STEP_COUNTER % 17) + 1
    else:
        episode = CURRENT_EPISODE
        step = STEP_COUNTER + 1
    
    STEP_COUNTER += 1
    return episode, step

def save_current_image_organized(pil_img, episode, step):
    """Save current input image in organized structure"""
    if not SAVE_RESULTS:
        return None
    
    # Use descriptive filename format: step_current_epXXX_stepYYY.jpg
    filename = f"{step}_current_ep{episode}_step{step}.jpg"
    
    # Save to each enabled method's directory
    saved_paths = []
    for method in ['rtcache', 'vinn', 'br']:
        if method in models_to_load:
            method_dir_name = get_method_directory_name(method)
            method_dir = Path(RESULTS_DIR) / method_dir_name / str(episode)
            method_dir.mkdir(exist_ok=True)
            
            image_path = method_dir / filename
            pil_img.save(image_path)
            saved_paths.append(str(image_path))
            print(f"[SAVE] Current image: {image_path}")
    
    return saved_paths

def save_action_vectors(episode, step, trajectories):
    """Save action vectors as JSON for each method"""
    if not SAVE_RESULTS:
        return
    
    # Save for each enabled method
    for method in ['rtcache', 'vinn', 'br']:
        if method in models_to_load:
            method_dir_name = get_method_directory_name(method)
            method_dir = Path(RESULTS_DIR) / method_dir_name / str(episode)
            method_dir.mkdir(exist_ok=True)
            
            # Create action data
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
            elif method == 'vinn' and f"{method}_trajectory" in trajectories:
                action_data["action_vector"] = trajectories[f"{method}_trajectory"][0] if trajectories[f"{method}_trajectory"] else None
                action_data["trajectory"] = trajectories[f"{method}_trajectory"]
            elif method == 'br' and f"{method}_trajectory" in trajectories:
                action_data["action_vector"] = trajectories[f"{method}_trajectory"][0] if trajectories[f"{method}_trajectory"] else None
                action_data["trajectory"] = trajectories[f"{method}_trajectory"]
            
            # Save action vector JSON with descriptive naming
            action_file = method_dir / f"{step}_action_vector_ep{episode}_step{step}.json"
            with open(action_file, 'w') as f:
                json.dump(action_data, f, indent=2)
            
            print(f"[SAVE] Action vector: {action_file}")

def save_retrieved_images_organized(episode, step, filtered_ids):
    """Save retrieved images in organized structure (for RT-Cache)"""
    if not SAVE_RESULTS or not LOAD_RTCACHE or not filtered_ids:
        return
    
    rtcache_dir_name = get_method_directory_name('rtcache')
    rtcache_dir = Path(RESULTS_DIR) / rtcache_dir_name / str(episode)
    rtcache_dir.mkdir(exist_ok=True)
    
    # Save retrieved images for RT-Cache with proper naming
    retrieved_count = 0
    
    for sample_idx, sample_id in enumerate(filtered_ids):
        # Parse sample_id to get episode and step info
        parts = sample_id.split("_")
        if len(parts) >= 3:
            retrieved_ep = parts[1]
            retrieved_step = parts[2]
            
            # Save all consecutive steps for this sample
            for base_step_offset in range(CONSECUTIVE_STEPS + 1):
                actual_step = int(retrieved_step) + base_step_offset
                raw_file = _raw_path(retrieved_ep, actual_step)
                
                if raw_file and os.path.exists(raw_file):
                    retrieved_count += 1
                    # Use naming format: step_retrieved_epXXX_stepYYY.jpg  
                    retrieved_filename = f"{step}_retrieved_ep{retrieved_ep}_step{actual_step}.jpg"
                    dest_path = rtcache_dir / retrieved_filename
                    shutil.copy2(raw_file, dest_path)
                    print(f"[SAVE] Retrieved image: {dest_path}")
    
    # Also save multiple samples if we have more candidates
    # This handles the case where RT-Cache retrieves multiple similar samples
    if len(filtered_ids) > 1:
        print(f"[INFO] Saved {retrieved_count} retrieved images from {len(filtered_ids)} similar samples")
        
        # Save metadata about all retrieved samples
        retrieval_metadata = {
            "episode": episode,
            "step": step,
            "retrieved_samples": [],
            "total_retrieved_images": retrieved_count
        }
        
        for sample_id in filtered_ids:
            parts = sample_id.split("_")
            if len(parts) >= 3:
                retrieval_metadata["retrieved_samples"].append({
                    "sample_id": sample_id,
                    "source_episode": parts[1],
                    "source_step": parts[2],
                    "consecutive_steps": CONSECUTIVE_STEPS + 1
                })
        
        # Save retrieval metadata
        metadata_file = rtcache_dir / f"{step}_retrieval_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(retrieval_metadata, f, indent=2)
        print(f"[SAVE] Retrieval metadata: {metadata_file}")

###############################################################################
# 6) VINN Predictor
###############################################################################
class VINNPredictor:
    def __init__(self, model_dir=VINN_MODEL_DIR, device=DEVICE, k=16):
        self.device = device
        self.k = k
        self.initialized = False
        
        print(f"🔄 Loading VINN model from {model_dir}...")
        
        try:
            self.encoder = resnet50(pretrained=False)
            self.encoder.fc = nn.Identity()
            
            encoder_path = f"{model_dir}/encoder.pth"
            if not os.path.exists(encoder_path):
                print(f"⚠️  VINN encoder not found at {encoder_path}")
                return
                
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            self.encoder = self.encoder.to(device)
            self.encoder.eval()
            
            self.database_embeddings = np.load(f"{model_dir}/database_embeddings.npy")
            self.database_actions = np.load(f"{model_dir}/database_actions.npy")
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.initialized = True
            print(f"✅ VINN loaded: {len(self.database_embeddings)} demonstrations")
        except Exception as e:
            print(f"❌ VINN initialization failed: {e}")
    
    def predict(self, pil_image):
        if not self.initialized:
            return None
        
        try:
            query_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_embedding = self.encoder(query_tensor).cpu().numpy()[0]
            
            distances = []
            for i, demo_embedding in enumerate(self.database_embeddings):
                dist = np.linalg.norm(query_embedding - demo_embedding)
                distances.append((dist, i))
            
            distances.sort(key=lambda x: x[0])
            top_k = distances[:self.k]
            
            weights = []
            actions = []
            
            for dist, idx in top_k:
                weight = np.exp(-dist)
                weights.append(weight)
                actions.append(self.database_actions[idx])
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            predicted_action = np.zeros_like(actions[0])
            for weight, action in zip(weights, actions):
                predicted_action += weight * action
            
            return predicted_action.tolist()
        except Exception as e:
            print(f"❌ VINN prediction failed: {e}")
            return None

###############################################################################
# 6) BehaviorRetrieval Predictor
###############################################################################
class BRPredictor:
    def __init__(self, model_dir=BR_MODEL_DIR, device=DEVICE):
        self.device = device
        self.initialized = False
        
        print(f"🔄 Loading BehaviorRetrieval model from {model_dir}...")
        
        try:
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 64)
            ).to(device)
            
            self.policy = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 7)
            ).to(device)
            
            policy_path = f"{model_dir}/policy_target_training.pth"
            if not os.path.exists(policy_path):
                print(f"⚠️  BR policy not found at {policy_path}")
                return
                
            self.policy.load_state_dict(torch.load(policy_path, map_location=device))
            
            self.visual_encoder.eval()
            self.policy.eval()
            self.initialized = True
            print("✅ BehaviorRetrieval model loaded")
        except Exception as e:
            print(f"❌ BehaviorRetrieval initialization failed: {e}")
    
    def predict(self, pil_image):
        if not self.initialized:
            return None
        
        try:
            image_resized = pil_image.resize((84, 84), Image.LANCZOS)
            image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                visual_features = self.visual_encoder(image_tensor)
                predicted_action = self.policy(visual_features).cpu().numpy()[0]
            
            return predicted_action.tolist()
        except Exception as e:
            print(f"❌ BehaviorRetrieval prediction failed: {e}")
            return None

###############################################################################
# Initialize predictors based on command-line arguments
###############################################################################
print("### Initializing predictors...")
vinn_predictor = None
br_predictor = None

if LOAD_VINN:
    print("   Loading VINN...")
    vinn_predictor = VINNPredictor()
else:
    print("   Skipping VINN (not requested)")

if LOAD_BR:
    print("   Loading BehaviorRetrieval...")
    br_predictor = BRPredictor()
else:
    print("   Skipping BehaviorRetrieval (not requested)")

if not LOAD_RTCACHE:
    print("   Skipping RT-Cache (not requested)")

print("### Predictor initialization complete.\n")

###############################################################################
# 7) Searching for raw images in final_images/raw/<episode>/<step> etc.
###############################################################################
def _raw_path(episode, step_i):
    """
    Check for step.jpg, step.png, 0-padded, etc.
    e.g. final_images/raw/<episode>/14.jpg or 14.png or 02.jpg.
    """
    ep_folder = os.path.join(RAW_BASE_DIR, str(episode))
    candidates = [
        os.path.join(ep_folder, f"{step_i}.jpg"),
        os.path.join(ep_folder, f"{step_i}.png"),
        os.path.join(ep_folder, f"{step_i:02d}.jpg"),
        os.path.join(ep_folder, f"{step_i:02d}.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

###############################################################################
# 6) gather_consecutive_actions + average
###############################################################################
def gather_consecutive_actions(sample_id: str, n_steps: int):
    """
    E.g. sample_id='test_1_14', if n_steps=3 => gather docs for 14,15,16,17
    """
    parts = sample_id.rsplit("_", 1)
    if len(parts) != 2:
        return []
    prefix, step_str = parts
    try:
        base_step = int(step_str)
    except ValueError:
        return []
    seq = []
    for offset in range(n_steps + 1):
        doc_id = f"{prefix}_{base_step + offset}"
        doc    = doc_cache.get(doc_id, None)
        if not doc:
            return []
        raw_act = doc.get("raw_action", ZERO_ACTION)
        if raw_act == ZERO_ACTION:
            return []
        # Debug logs
        print("#### Selected Docs", doc)
        seq.append(raw_act)
        print("#### SEQ", seq)
    return seq

def compute_average_trajectory(candidate_list):
    if not candidate_list:
        return []
    arr = np.array(candidate_list)
    avg = arr.mean(axis=0)
    return avg.tolist()

###############################################################################
# 7) LlamaIndex Qdrant Search
###############################################################################
def qdrant_search_image(pil_image: Image.Image, topk=10):
    print("##### Start LlamaIndex Search (image)")
    print("#### BEFORE Vector SEARCH")
    emb = get_image_embedding(pil_image)
    query_list = emb.float().squeeze(0).tolist()
    vs_query = VectorStoreQuery(query_embedding=query_list, similarity_top_k=topk)
    results: VectorStoreQueryResult = image_store.query(vs_query)
    print("#### DONE Vector SEARCH")
    print("##### Done LlamaIndex Search (image)")

    out = []
    if results.nodes:
        for i, node in enumerate(results.nodes):
            sim_val = results.similarities[i] if results.similarities else 0.0
            meta = node.metadata or {}
            c_logical = meta.get("logical_id", "")
            out.append({"logical_id": c_logical, "score": sim_val})

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

###############################################################################
# 8) If you want centroid-based "closest dataset" detection
###############################################################################
def predict_closest_dataset_with_centroids(query_tensor):
    qvec = query_tensor.squeeze(0).tolist()
    results = qdrant_client.search(
        collection_name=CENTROID_COLL_NAME,
        query_vector=qvec,
        limit=1,
        with_payload=True,
        with_vectors=False
    )
    if not results:
        return "unknown"
    return results[0].payload.get("dataset_name", "unknown")

###############################################################################
# 9) parse episode + step from doc_id
###############################################################################
def parse_episode_and_step(sample_id):
    """
    'test_1_14' => ep='1', st='14'
    """
    parts = sample_id.split("_")
    if len(parts) < 3:
        return None, None
    return parts[1], parts[2]

###############################################################################
# 10) Save logic: user (current) image & retrieved images
###############################################################################
def save_current_image(pil_img, ep, st):
    """
    e.g. "1_current_ep9999_step1.jpg"
    """
    filename = f"{call_count}_current_ep{ep}_step{st}.jpg"
    outpath  = os.path.join(CURRENT_LOG_DIR, filename)
    pil_img.save(outpath)
    print(f"[SAVE] => {outpath}")
    return outpath

def save_retrieved_images_for_consecutive_steps(call_idx, user_ep, user_st, base_ep, base_step):
    """
    If base_step=14, and CONSECUTIVE_STEPS=3 => store steps 14,15,16,17
    e.g. "1_retrieved_ep1_step14.jpg"
         "1_retrieved_ep1_step15.jpg"
         "1_retrieved_ep1_step16.jpg"
         "1_retrieved_ep1_step17.jpg"
    """
    base_step = int(base_step)
    for offset in range(CONSECUTIVE_STEPS + 1):
        real_step = base_step + offset
        rawfile   = _raw_path(base_ep, real_step)
        if not rawfile:
            print(f"[WARN] No raw image found: epi={base_ep} step={real_step}")
            continue
        new_name = f"{call_idx}_retrieved_ep{base_ep}_step{real_step}.jpg"
        dest     = os.path.join(CURRENT_LOG_DIR, new_name)
        shutil.copy2(rawfile, dest)
        print(f"[SAVE] => {dest}")

###############################################################################
# 11) Flask: /pipeline
###############################################################################
@app.route("/pipeline", methods=["POST"])
def pipeline():
    global call_count, CURRENT_LOG_DIR

    t0 = time.time()
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    print("##### START pipeline")
    file   = request.files["file"]
    prompt = request.form.get("instruction", "")
    option = request.form.get("option", "both")

    print("##### Received IMAGE")

    # 1) Rotate folder every 17 calls
    call_count += 1
    if call_count % EPISODE_LENGTH == 1 and call_count > 1:
        CURRENT_LOG_DIR = _new_log_dir()
        print(f"[INFO] Rotating logs folder => {CURRENT_LOG_DIR}")

    # 2) Preprocess
    pil_image = Image.open(file).convert("RGB")
    pil_image = center_crop_to_square(pil_image)
    pil_image = pil_image.resize((224,224), Image.Resampling.LANCZOS)
    print("##### DONE Image Processing")

    # Get episode and step numbers for organized saving
    episode, step = get_episode_and_step()
    
    # Save the user input image in organized structure
    save_current_image_organized(pil_image, episode, step)
    
    # Also save in old format for compatibility
    user_ep = episode
    user_st = step
    save_current_image(pil_image, user_ep, user_st)

    t1 = time.time()

    # 3) Embedding
    print("##### Start Get Embedding")
    image_features = None
    if option in ["image","both"]:
        image_features = get_image_embedding(pil_image)
    print("##### Done Embedding")

    # Optionally do text embedding:
    # text_features = ...

    # Optionally find closest dataset
    best_dataset = "unknown"
    if image_features is not None:
        best_dataset = predict_closest_dataset_with_centroids(image_features)
        print("[INFO] Closest dataset is:", best_dataset)

    t2 = time.time()

    # 4) Qdrant search (only if RT-Cache is enabled)
    image_hits = []
    if LOAD_RTCACHE and image_features is not None:
        llama_image_results = qdrant_search_image(pil_image, topk=DB_LIMIT_NUM)
        for item in llama_image_results:
            image_hits.append({
                "payload": {"logical_id": item["logical_id"]},
                "score":   item["score"]
            })

    t3 = time.time()

    # 5) Filter + gather consecutive steps (only if RT-Cache is enabled)
    final_ids     = []
    final_actions = []
    if LOAD_RTCACHE:
        for hit in image_hits:
            sample_id = hit["payload"].get("logical_id", None)
            if not sample_id:
                continue

            # gather consecutive
            seq = gather_consecutive_actions(sample_id, CONSECUTIVE_STEPS)
            if seq:
                final_ids.append(sample_id)
                final_actions.append(seq)
                if len(final_ids) >= NUM_CANDIDATES:
                    break

    t4 = time.time()

    # 6) Average trajectory (only if RT-Cache is enabled)
    averaged_trajectory = compute_average_trajectory(final_actions) if LOAD_RTCACHE else []

    # 7) Get predictions from VINN and BehaviorRetrieval
    print("##### Getting predictions from enabled models...")
    vinn_trajectory = []
    br_trajectory = []
    rtcache_trajectory = averaged_trajectory
    
    if LOAD_VINN and vinn_predictor and vinn_predictor.initialized:
        vinn_action = vinn_predictor.predict(pil_image)
        if vinn_action is not None:
            vinn_trajectory = [vinn_action]  # Wrap single prediction in list like averaged_trajectory
        print(f"[RESULTS] VINN trajectory: {vinn_trajectory}")
    
    if LOAD_BR and br_predictor and br_predictor.initialized:
        br_action = br_predictor.predict(pil_image)
        if br_action is not None:
            br_trajectory = [br_action]  # Wrap single prediction in list like averaged_trajectory
        print(f"[RESULTS] BehaviorRetrieval trajectory: {br_trajectory}")
    
    if LOAD_RTCACHE:
        print(f"[RESULTS] RT-Cache trajectory: {rtcache_trajectory}")
    
    # 8) Save results in organized structure
    if SAVE_RESULTS:
        # Prepare trajectory data for saving
        trajectory_data = {
            "filtered_ids": final_ids,
            "episode": episode,
            "step": step
        }
        
        if LOAD_RTCACHE:
            trajectory_data["rtcache_trajectory"] = rtcache_trajectory
            trajectory_data["rtcache"] = rtcache_trajectory
        if LOAD_VINN:
            trajectory_data["vinn_trajectory"] = vinn_trajectory
        if LOAD_BR:
            trajectory_data["br_trajectory"] = br_trajectory
        
        # Save action vectors as JSON
        save_action_vectors(episode, step, trajectory_data)
        
        # Save retrieved images (for RT-Cache)
        save_retrieved_images_organized(episode, step, final_ids)
    
    # 9) Save retrieved images for *all* consecutive steps (old format for compatibility)
    for fid in final_ids:
        ep_str, step_str = parse_episode_and_step(fid)
        if ep_str is None or step_str is None:
            continue
        save_retrieved_images_for_consecutive_steps(
            call_idx = call_count,
            user_ep  = user_ep,
            user_st  = user_st,
            base_ep  = ep_str,
            base_step= step_str
        )

    # Return JSON with enabled methods only (all using trajectory format)
    response = {
        "closest_dataset": best_dataset,
        "enabled_models": models_to_load,
        "episode": episode,
        "step": step,
        "save_results": SAVE_RESULTS
    }
    
    # Determine which trajectory to use as averaged_trajectory (priority: RT-Cache > VINN > BR)
    averaged_trajectory_output = []
    if LOAD_RTCACHE and rtcache_trajectory:
        averaged_trajectory_output = rtcache_trajectory
    elif LOAD_VINN and vinn_trajectory:
        averaged_trajectory_output = vinn_trajectory
    elif LOAD_BR and br_trajectory:
        averaged_trajectory_output = br_trajectory
    
    # Always include averaged_trajectory in response
    response["averaged_trajectory"] = averaged_trajectory_output
    
    # Add RT-Cache results if enabled
    if LOAD_RTCACHE:
        response.update({
            "filtered_ids": final_ids,
            "rtcache_trajectory": rtcache_trajectory
        })
    
    # Add VINN results if enabled (same format as averaged_trajectory)
    if LOAD_VINN:
        response["vinn_trajectory"] = vinn_trajectory
    
    # Add BehaviorRetrieval results if enabled (same format as averaged_trajectory)
    if LOAD_BR:
        response["br_trajectory"] = br_trajectory

    t5 = time.time()
    print("[TIMING] parse=%.3fs, embed=%.3fs, search=%.3fs, filter=%.3fs, total=%.3fs" %
          (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t0))

    return jsonify(response), 200

###############################################################################
# 12) /gallery + /static_images
###############################################################################
@app.route("/gallery", methods=["GET"])
def gallery():
    """
    Displays the images in CURRENT_LOG_DIR
    """
    if not os.path.exists(CURRENT_LOG_DIR):
        return "<h3>No logs folder found.</h3>"

    files = [f for f in os.listdir(CURRENT_LOG_DIR) if f.lower().endswith(".jpg")]
    files.sort()
    html = "<html><head><title>Image Gallery</title></head><body>"
    html += f"<h2>Current logs folder: {CURRENT_LOG_DIR}</h2>"
    for f in files:
        html += f"<p>{f}</p>"
        html += f'<img src="/static_images/{f}" style="max-width:400px;"><br><br>'
    html += "</body></html>"
    return html

@app.route("/static_images/<filename>")
def serve_image(filename):
    return send_from_directory(CURRENT_LOG_DIR, filename)

###############################################################################
# 13) Offline Mode (Optional)
###############################################################################
def load_single_episode(tfrecord_path):
    return None

def run_offline_mode():
    pass

###############################################################################
# Main
###############################################################################
def main():
    if RUN_SERVER:
        host = args.host if hasattr(args, 'host') else config.server.retrieval_host
        port = args.port if hasattr(args, 'port') else config.server.retrieval_port
        app.run(host=host, port=port, debug=False)
    else:
        run_offline_mode()

if __name__ == "__main__":
    main()
