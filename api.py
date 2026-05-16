# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import warnings
import time
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

app = FastAPI(
    title="Silent Face Anti-Spoofing API",
    description="Face liveness detection API - classifies faces as real or fake (deepfake/print/screen/mask)",
    version="1.0.0",
)

MODEL_DIR = os.environ.get("MODEL_DIR", "./resources/anti_spoof_models")
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))

model_test = None
image_cropper = None


@app.on_event("startup")
def startup():
    global model_test, image_cropper
    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()


def predict_image(image: np.ndarray) -> dict:
    """Run anti-spoofing prediction on an image (BGR numpy array)."""
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    for model_name in os.listdir(MODEL_DIR):
        if not model_name.endswith(".pth"):
            continue
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        test_speed += time.time() - start

    label = np.argmax(prediction)
    score = float(prediction[0][label] / 2)
    is_real = bool(label == 1)

    return {
        "is_real": is_real,
        "label": int(label),
        "score": round(score, 4),
        "prediction_time_s": round(test_speed, 4),
        "face_bbox": {
            "x": image_bbox[0],
            "y": image_bbox[1],
            "width": image_bbox[2],
            "height": image_bbox[3],
        },
    }


@app.post("/check")
async def check_face(image: UploadFile = File(...)):
    """
    Check if a face in the uploaded image is real or fake.

    Accepts JPEG/PNG image. Returns prediction result with score.
    """
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        result = predict_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
