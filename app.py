from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

pipeline = joblib.load("mushroom_pipeline.pkl")

app = FastAPI(title="Mushroom Classifier API")

class MushroomInput(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: MushroomInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    
    return {"prediction": prediction}