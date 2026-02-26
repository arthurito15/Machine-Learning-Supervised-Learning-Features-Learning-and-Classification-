# Konkobo Ulrich Arthur p2513439 & PELLOIS Guillaume p2102360 & Issoumaila Fomba p2512887

import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile
import os
import io

# Le nom du fichier que la fonction pipeline_generation_train_test_split a créé
PIPELINE_FILE = 'final_pipeline_q8.pkl' 

# Fonction pour charger le pipeline
def load_pipeline():
    """Charge le pipeline depuis le fichier pickle."""
    with open(PIPELINE_FILE, 'rb') as file:
        pipeline = pickle.load(file)
        return pipeline

# Chargement du modèle au démarrage de l'API
model_pipeline = load_pipeline()

app = FastAPI(
    title="Credit Scoring API",
    description="API pour prédire le risque de crédit en utilisant un pipeline Scikit-learn optimisé (Q8)."
)

# Test:
# fastapi run api.py
# curl -X POST -F file=@credit_scoring.csv http://localhost:8000/predict/
@app.post("/predict/")
def predict_credit_risk(file: UploadFile = File(...)):
    if model_pipeline is None:
        return {"error": "Le pipeline de modélisation n'est pas prêt. Vérifiez le chargement."}
        
    #data_df = pd.read_csv(file.file, sep=";").drop("Status", axis=1)
    data_df = pd.read_csv(file.file, sep=";").drop("Status", axis=1)
    
    # Prédiction (classe 0 ou 1)
    prediction = model_pipeline.predict(data_df)[0]
        
    # Probabilités [proba_classe_0 (refus), proba_classe_1 (accordé)]
    probabilities = model_pipeline.predict_proba(data_df)[0]
        
    # Formatage de la réponse
    return {
            "prediction_classe": int(prediction),
            "interpretation": "Crédit accordé (1)" if prediction == 1 else "Crédit refusé (0)",
            "probability_refused_0": round(probabilities[0], 4),
            "probability_accorded_1": round(probabilities[1], 4),
    }
    




