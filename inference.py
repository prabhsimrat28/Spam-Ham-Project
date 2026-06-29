from model import mymodel
import torch
from preprocess import preprocess_data
from schema.prediction_response import PredictionResponse
import torch.nn.functional as F
model = mymodel()
model.load_state_dict(torch.load("models/spam_classifier.pth", map_location=torch.device('cpu')))
model.eval()

def predict(text:str):

    encoded = preprocess_data(text)
    
    encoded = torch.tensor(encoded,dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        logits = model(encoded)
        probabilities = F.softmax(logits, dim=1)
    
    prediction = torch.argmax(probabilities, dim=1).item()  # Get the predicted class index

    labels = ["Ham", "Spam"]
    predicted_result = labels[prediction]
    
    return PredictionResponse( 
        predicted_result=predicted_result, 
        confidence=probabilities[0, prediction].item(),
        class_probabilities={ "Ham": probabilities[0, 0].item(), "Spam": probabilities[0, 1].item(), },
        )