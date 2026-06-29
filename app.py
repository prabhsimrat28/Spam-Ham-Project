from fastapi import FastAPI,Path,HTTPException,Query
import json
from fastapi.responses import JSONResponse
from schema.prediction_response import PredictionResponse
from inference import predict
from schema.user_input import SMStext
app=FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to Text Classifiction into Spam/Ham made by Prabhsimrat Singh"}

@app.get("/about")
def explain():
    return { "message": ( "This project is a simple SMS classification model that classifies messages as Spam or Ham." "First, the text is cleaned by converting it to lowercase, removing punctuation, and tokenizing it." "Then, a vocabulary is built from the training dataset, and each unique word is assigned an integer." "Next, embeddings of dimension 100 are created using PyTorch Embeddings." "Finally, an LSTM model is used to process the textual data sequentially." ) }


@app.post("/predict",response_model=PredictionResponse)
def predict_spam(data:SMStext):
    predicted=predict(data.text)
    return predicted

