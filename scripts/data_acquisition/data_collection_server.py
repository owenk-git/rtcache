
#!/usr/bin/env python3

### DEBUG mode working well
### MODE=debug DEBUG_EPISODE=27 python scripts/data_acquisition/data_collection_server.py
# server.py  – FINAL (May-03-2025)
# ------------------------------------------------------------------
# DEBUG : 17-step scripted loop, auto-counter, no extra fields.
# TEST  : Retrieval pipeline (single embedding call).
# ------------------------------------------------------------------

import os, time, base64, shutil, requests, json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from pymongo import MongoClient
from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult

# Add config path
sys.path.append(str(Path(__file__).parent.parent.parent / "config"))
from rt_cache_config import get_config

# ───────── Global configuration ───────────────────────────────────
config = get_config()

# Experiment settings
MODE             = config.experiment.collection_mode
DEBUG_EPISODE    = config.experiment.debug_episode
RAW_BASE         = "final_images/raw"
LOGS_BASE        = config.paths.log_dir
RUN_SERVER       = True
REMOTE_EMBEDDING_URL = config.server.embedding_url

# Dataset settings
DATASET_NAME      = config.dataset.dataset_name

# Database settings
MONGO_URL         = config.database.mongo_url
DB_NAME           = config.database.mongo_db
ID_FIELD          = "id"
COLL_NAME         = DATASET_NAME + "_collection"
QDRANT_HOST       = config.database.qdrant_host
QDRANT_PORT       = config.database.qdrant_port
QDRANT_COLLECTION = "image_collection_" + DATASET_NAME

# Retrieval settings
DB_LIMIT_NUM      = 50
NUM_CANDIDATES    = config.retrieval.num_candidates
CONSECUTIVE_STEPS = config.retrieval.consecutive_steps
ZERO_ACTION       = config.experiment.zero_action

for p in (RAW_BASE, LOGS_BASE):
    os.makedirs(p, exist_ok=True)

app = Flask(__name__)

# ───────── DEBUG step counter ──────────────────────────────────────
debug_counter = 1            # counts 1 … 17

# ───────── Franka canned plan (full lookup) ───────────────────────
def get_action_vector(i: int, epi: str):
    # helper lambdas
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

# ───────── Helpers (cropping, embedding, qdrant) ──────────────────
def center_crop_224(p: Image.Image) -> Image.Image:
    w,h=p.size; s=min(w,h)
    return p.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)).resize((224,224))

def decode_b64_tensor(b64:str)->torch.Tensor:
    return torch.load(BytesIO(base64.b64decode(b64)), map_location="cpu")

def send_image_for_embedding(pil:Image.Image)->torch.Tensor:
    buf=BytesIO(); pil.save(buf,"JPEG"); buf.seek(0)
    r=requests.post(REMOTE_EMBEDDING_URL,
        files={"file":("img.jpg",buf,"image/jpeg")},
        data={"instruction":"","option":"image"}, timeout=300)
    r.raise_for_status()
    return decode_b64_tensor(r.json()["image_features"])

# TEST-only DB + Qdrant boot
if MODE=="test":
    mongo  = MongoClient(MONGO_URL)[DB_NAME][COLL_NAME]
    doc_cache = {d[ID_FIELD]:d for d in mongo.find({}, {ID_FIELD:1,"raw_action":1,"_id":0})}
    qdrant_client=QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)
    image_store  = QdrantVectorStore(client=qdrant_client,
                    collection_name=QDRANT_COLLECTION, content_key="logical_id")
    run_ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FOLDER=os.path.join(LOGS_BASE,run_ts); IMAGES_FOLDER=os.path.join(LOG_FOLDER,"images")
    os.makedirs(IMAGES_FOLDER,exist_ok=True)

def qdrant_search_by_emb(v,k=DB_LIMIT_NUM):
    res=image_store.query(VectorStoreQuery(query_embedding=v, similarity_top_k=k))
    return sorted([{"logical_id":n.metadata.get("logical_id",""),"score":res.similarities[i]}
                   for i,n in enumerate(res.nodes or [])],
                   key=lambda x:x["score"], reverse=True)

def gather_consecutive(sid,n):
    pfx,idx=sid.rsplit("_",1); base=int(idx); seq=[]
    for off in range(n+1):
        d=doc_cache.get(f"{pfx}_{base+off}")
        if not d: return []
        act=d.get("raw_action",ZERO_ACTION)
        if act==ZERO_ACTION: return []
        seq.append(act)
    return seq

# ───────── /pipeline endpoint ──────────────────────────────────────
@app.route("/pipeline", methods=["POST"])
def pipeline():
    global debug_counter

    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    pil = center_crop_224(Image.open(request.files["file"]).convert("RGB"))

    # ---- DEBUG branch (auto 17-step loop) -------------------------
    if MODE=="debug":
        if debug_counter>17:
            return jsonify({"done":True}), 200

        step_idx=debug_counter; debug_counter+=1
        ep_dir=os.path.join(RAW_BASE,DEBUG_EPISODE)
        os.makedirs(ep_dir,exist_ok=True)
        pil.save(os.path.join(ep_dir,f"{step_idx:02d}.jpg"))

        action=get_action_vector(step_idx,DEBUG_EPISODE)
        return jsonify({"action":action,"averaged_trajectory":[action]}),200

    # ---- TEST branch ---------------------------------------------
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    up_name=f"upload_{ts}.jpg"
    pil.save(os.path.join(IMAGES_FOLDER,up_name))

    emb=send_image_for_embedding(pil).float().squeeze(0).tolist()
    hits=qdrant_search_by_emb(emb,DB_LIMIT_NUM)

    filt,traj=[],[]
    for h in hits:
        sid=h["logical_id"]
        seq=gather_consecutive(sid,CONSECUTIVE_STEPS)
        if seq:
            filt.append(sid); traj.extend(seq)
            if len(filt)>=NUM_CANDIDATES: break

    return jsonify({"filtered_ids":filt,"averaged_trajectory":traj}),200

# ───────── tiny gallery (debug-only) ───────────────────────────────
@app.route("/gallery")
def gallery():
    if not os.path.exists(RAW_BASE):
        return "<h3>No debug images saved yet.</h3>"
    files=sorted(f for f in os.listdir(os.path.join(RAW_BASE,DEBUG_EPISODE))
                 if f.endswith(".jpg"))
    html="<html><body>"+ "".join(
        f"<p>{f}</p><img src='/static/{f}' style='max-width:300px'><br>"
        for f in files)+"</body></html>"
    return html

@app.route("/static/<fname>")
def send_debug(fname):
    return send_from_directory(os.path.join(RAW_BASE,DEBUG_EPISODE),fname)

# ───────── main ────────────────────────────────────────────────────
if __name__=="__main__":
    print(f"# server.py in {MODE.upper()} mode — port 5002")
    if RUN_SERVER:
        app.run(host="0.0.0.0", port=5002, debug=False)
