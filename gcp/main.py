from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()

model = None
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

BUCKET_NAME = "sharan-tf-models1"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

@app.on_event("startup")
def load_model():
    global model
    if model is None:
        model_path = "/tmp/tomatoes.h5"
        download_blob(BUCKET_NAME, "models/tomatoes.h5", model_path)
        model = tf.keras.models.load_model(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    image = Image.open(file.file).convert("RGB").resize((256, 256))
    image = np.array(image) / 255.0
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Ensure it listens on port 8080
