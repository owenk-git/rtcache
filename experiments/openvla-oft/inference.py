#!/usr/bin/env python3
# server_custom_vla.py  – 2025-05-05
# FastAPI endpoint that runs your *custom-trained* OpenVLA model.

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import io, json, os
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# ─────── Configuration ────────────────────────────────────────────
CKPT_DIR = os.getenv("OPENVLA_CHECKPOINT_DIR", "./checkpoints/openvla-7b-franka-finetuned")
DTYPE    = torch.bfloat16          # keep the same dtype used in training
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"

# ─────── Load processor & model ───────────────────────────────────
# Both the tokenizer/processor and the model live in CKPT_DIR
processor = AutoProcessor.from_pretrained(
    CKPT_DIR,
    trust_remote_code=True         # uses your custom *prismatic* classes
)

vla = AutoModelForVision2Seq.from_pretrained(
    CKPT_DIR,
    attn_implementation="flash_attention_2",  # keep if flash-attn is installed
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(DEVICE).eval()                # eval() disables dropout, etc.

# ─────── Small image helper ───────────────────────────────────────
def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left  = (w - side) / 2
    top   = (h - side) / 2
    right = (w + side) / 2
    bot   = (h + side) / 2
    return img.crop((left, top, right, bot))

# ─────── FastAPI app ──────────────────────────────────────────────
app = FastAPI()

@app.post("/openvla-predict")
async def predict(
    file: UploadFile = File(...),
    instruction: str = Form(...),
    option: str = Form(...)
):
    """
    Form-data:
      • file         – image (any format Pillow can read)
      • instruction  – natural-language prompt (e.g. "pick up the block")
      • option       – any extra metadata you want echoed back

    Returns {"action": [...], "option_received": "..."}
    """
    # 1) Load & pre-process the image
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = center_crop_to_square(img).resize((224, 224), Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2) Build the prompt (follow the same IO pattern as training)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # 3) Tokenise / encode image + text
    try:
        inputs = processor(prompt, img).to(DEVICE, dtype=DTYPE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processor error: {e}")

    # 4) Run the model
    try:
        # 'bridge_orig' was your un-normalisation key during finetuning;
        # change it if you trained with a different key.
        action = vla.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    # 5) Ensure JSON-serialisable output
    if isinstance(action, torch.Tensor):
        action = action.cpu().tolist()
    else:
        # fall back to string if model returns something exotic
        try:
            json.dumps(action)
        except Exception:
            action = str(action)

    return {"action": action, "option_received": option}

# ─────── Entry-point for local testing ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9010"))
    uvicorn.run("server_custom_vla:app", host=host, port=port, log_level="info")
