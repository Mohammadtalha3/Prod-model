from transformers import DistilBertForSequenceClassification
from safetensors.torch import load_file  # for loading .safetensors
import torch

def load_finetuned_model():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")  # Load model architecture
    weights = load_file("D:/prod_model/app/data/model.safetensors")  # Load weights from safetensors file
    model.load_state_dict(weights)  # Set the weights to the model
    model.eval()  # Set to evaluation mode
    return model
