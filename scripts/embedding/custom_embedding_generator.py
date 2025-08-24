#!/usr/bin/env python3
# custom_embedding_generator.py  – 2025-05-04 (rev-D)
# Re-creates a fresh Qdrant collection + Mongo collection, uploads a
# user-selectable subset of episodes, then prints a quick sanity check:
#   • total #points in Qdrant and #docs in Mongo
#   • one example entry from each.

from __future__ import annotations
import argparse, base64, uuid
from pathlib import Path
from typing import List
from io import BytesIO

import requests, torch
from PIL import Image
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest, models as qmodels

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
from rt_cache_config import get_config

# ───────── Configuration ────────────────────────────────────────────
config = get_config()

ROOT_DIR            = Path(config.paths.rt_cache_raw_dir)
REMOTE_EMBEDDING_URL = config.server.embedding_url

MONGO_URI           = config.database.mongo_url
MONGO_DB            = config.database.mongo_db
MONGO_COL           = config.dataset.dataset_name + "_collection"

QDRANT_HOST         = config.database.qdrant_host
QDRANT_PORT         = config.database.qdrant_port
QDRANT_COL          = f"image_collection_{config.dataset.dataset_name}_default"
EMB_DIM             = config.retrieval.openvla_dim
DIST_METRIC         = config.retrieval.distance_metric

# Import action generator system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_acquisition'))
from action_generators import create_action_generator

# Initialize action generator using configuration
action_generator = create_action_generator(
    config.experiment.action_generator_type,
    config_path=config.experiment.action_config_path
)
OPT_MAP = action_generator.get_episode_options()

# ───────── Helpers ──────────────────────────────────────────────────
def get_image_embedding(img: Image.Image) -> torch.Tensor:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    resp = requests.post(
        REMOTE_EMBEDDING_URL,
        files={"file": ("image.jpg", buf, "image/jpeg")},
        data={"instruction": "", "option": "image"},
        timeout=300,
    )
    resp.raise_for_status()
    tensor = torch.load(BytesIO(base64.b64decode(resp.json()["image_features"])),
                        map_location="cpu")
    return tensor.squeeze(0)

def get_action_vector(step_idx: int, episode_idx: str) -> List[float]:
    """
    Generate action vector using the modular action generator system.
    
    Users can customize the action generation strategy by:
    1. Modifying the action_generator initialization above
    2. Creating custom ActionGenerator subclasses
    3. Using configuration files (for ConfigurableActionGenerator)
    """
    return action_generator.get_action_vector(step_idx, episode_idx)

# ───────── Main upload routine ─────────────────────────────────────
def upload(option: str):
    episodes = OPT_MAP[option]

    # DB connections
    mongo = MongoClient(MONGO_URI)[MONGO_DB][MONGO_COL]
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    # Recreate Qdrant collection (drop -> create)
    if any(c.name == QDRANT_COL for c in qdrant.get_collections().collections):
        qdrant.delete_collection(collection_name=QDRANT_COL)
    qdrant.create_collection(
        collection_name=QDRANT_COL,
        vectors_config=rest.VectorParams(size=EMB_DIM, distance=DIST_METRIC),
    )

    # Clear Mongo
    mongo.delete_many({})

    # Upload loop
    for eid in episodes:
        folder = ROOT_DIR / str(eid)
        if not folder.is_dir():
            print(f"[WARN] {folder} missing – skip."); continue

        for step in range(1, 18):
            # img_file = folder / f"{step}.jpg"
            # if not img_file.is_file():
            #     continue
            img_file = folder / f"{step}.jpg"          # plain
            if not img_file.is_file():                 # try zero-pad
                img_file = folder / f"{step:02d}.jpg"
            if not img_file.is_file():                 # still missing → skip
                continue            
            try:
                vec = get_image_embedding(Image.open(img_file).convert("RGB"))
                if vec.shape[0] != EMB_DIM:
                    raise ValueError("dim mismatch")
            except Exception as e:
                print(f"[ERR] {img_file.name}: {e}"); continue

            act = get_action_vector(step, str(eid))
            doc_id = f"test_{eid}_{step}"

            # Mongo insert
            mongo.insert_one({
                "id": doc_id,
                "filename": img_file.name,
                "step_idx": step,
                "raw_action": act,
                "norm_action": act,
                "episode_idx": f"{eid}/",
                "point_idx": 0,
                "total_steps_in_episode": 0,
                "dataset_name": "test",
                "text": "",
            })

            # Qdrant upsert
            qdrant.upsert(
                collection_name=QDRANT_COL,
                points=[rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "logical_id": doc_id,
                        "step_idx": step,
                        "dataset_name": "test",
                        "episode_idx": f"{eid}/",
                        "point_idx": 0,
                        "text": "",
                    },
                )],
            )
        print(f"[OK] episode {eid} done.")

    # ─── Quick sanity check ────────────────────────────────
    total_qdrant = qdrant.count(QDRANT_COL, exact=True).count
    total_mongo  = mongo.count_documents({})

    # example point/doc
    ex_point     = qdrant.scroll(QDRANT_COL, limit=1, offset=0)[0][0]
    ex_doc       = mongo.find_one({}, {"_id": 0})

    print("\n─── Sanity check ──────────────────────────────────")
    print(f"Qdrant points : {total_qdrant}")
    print(f"Mongo docs    : {total_mongo}\n")

    print("Example Qdrant payload:")
    print(ex_point.payload)

    print("\nExample Mongo document:")
    print(ex_doc)
    print("───────────────────────────────────────────────────")

# ───────── CLI ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Rebuild test embeddings, then show counts + example.")
    parser.add_argument("--option",
                        required=True,
                        choices=list(OPT_MAP.keys()),
                        help="Subset of episodes to upload.")
    args = parser.parse_args()
    upload(args.option)

if __name__ == "__main__":
    main()
