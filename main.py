import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

app = FastAPI()
app.mount("/web", StaticFiles(directory="web"), name="web")

model = tf.keras.models.load_model('model/efficientnet_best.keras')

with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

with open("model/disease_info.json", "r") as f:
    disease_info = json.load(f)


def predict_image(img: Image.Image, top_k=3):
    img = img.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    preds = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:top_k]
    results = []

    for i in top_indices:
        class_name = class_names[i]
        species, disease = class_name.split("___", 1) if "___" in class_name else (class_name, "Unknown")
        species = species.replace("_", " ").strip()
        disease = disease.replace("_", " ").strip()
        status = "Healthy" if disease.lower() == "healthy" else disease

        info = disease_info.get(class_name, {})
        
        results.append({
            "species": species,
            "status": status,
            "confidence": round(float(preds[i]) * 100, 2),
            "symptoms": info.get("symptoms", "N/A"),
            "causes": info.get("causes", "N/A"),
            "cures": info.get("cures", "N/A")
        })

    return results


@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("web/index.html", "r") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    return {"predictions": predict_image(img)}