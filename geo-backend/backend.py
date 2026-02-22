import io
import numpy as np
import torch
import joblib
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
app = FastAPI()

# 1. CORS Middleware (Crucial for connecting Next.js to FastAPI)
app.add_middleware(
    CORSMiddleware,
    # Next.js localhost port
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

MODEL_NAME = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip.eval()

#Load trained SVM + label encoder 
clf = joblib.load("geo_svm_calibrated.joblib")
le  = joblib.load("label_encoder.joblib")

#embedding helper function
@torch.no_grad()
def embed_image(image: Image.Image) -> np.ndarray:
   #Return a normalized CLIP embedding (1 × 512) as a numpy array
    inputs = processor(images=image, return_tensors="pt").to(device)
    feats = clip.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()



@app.post("/predict")
async def receive_image(file: UploadFile = File(...)):
    # 2. Read the raw binary data
    data = await file.read()
      
        
    try:
        image = Image.open(io.BytesIO(data))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        return {"status": "error", "message": f"Could not open image: {e}"}

        
    embedding = embed_image(image)           # shape (1, 512)
    probs = clf.predict_proba(embedding)[0]  # shape (n_classes,)
        # ---------------------------------------------------------
        
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [
            {"country": le.inverse_transform([i])[0], "confidence": round(float(probs[i]), 4)}
            for i in top5_idx
        ]

    return {
            "status": "success",
            "country": top5[0]["country"],
            "confidence": top5[0]["confidence"],
            "top5": top5,
        }
        

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>GeoGuessr API is running!</h1>"