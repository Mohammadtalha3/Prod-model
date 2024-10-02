from fastapi import FastAPI
from pydantic import BaseModel
from load_model import load_finetuned_model
from transformers import DistilBertTokenizer
import torch
from prometheus_fastapi_instrumentator import Instrumentator
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import time

# Initialize FastAPI app
app = FastAPI()

# Instrumentation for Prometheus
Instrumentator().instrument(app).expose(app)

# Load the IMDB dataset
data = load_dataset('imdb')

# Load the model and tokenizer
model = load_finetuned_model("D:\prod_model\model.safetensors")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")



# Define a request body schema
class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Model is ready to serve"}

@app.post("/predict")
def predict(request: TextRequest):
    start_time = time.time()
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    inference_time = time.time() - start_time

    mlflow.start_run()  # Start a new run for the prediction
    mlflow.log_param("input_text", request.text)  # Log the input text
    mlflow.log_metric("prediction", prediction)   # Log the prediction
    mlflow.log_metric("inference_time", inference_time)  # Log inference time
    mlflow.end_run()  # End the run

    return {"prediction": prediction, "inference_time": inference_time}

@app.post("/evaluate")
def evaluate():
    # Load the test dataset
    dataset = load_dataset('imdb')['test']

    # Batch evaluation
    batch_size = 16
    predictions = []
    labels = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)


        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            labels.extend(batch['label'])
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    # Log metrics to MLflow
    mlflow.start_run()
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.end_run()

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }


@app.post("/batch_evaluate")
def batch_evaluate():
    # Load the 2k test dataset
    test_data = load_dataset('imdb')['test'].select(range(2000))  # Select first 2000 rows

    predictions = []
    actual_labels = []
    
    batch_size = 16  # Set the batch size for evaluation

    for i in range(0, len(test_data), batch_size):
        batch = test_data.select(range(i, min(i + batch_size, len(test_data))))
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).tolist()

        predictions.extend(batch_predictions)
        actual_labels.extend(batch['label'])

    # Calculate the metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, average='weighted')
    recall = recall_score(actual_labels, predictions, average='weighted')
    f1 = f1_score(actual_labels, predictions, average='weighted')

    # Log the evaluation metrics to MLflow
    mlflow.start_run()
    mlflow.log_metric("batch_accuracy", accuracy)
    mlflow.log_metric("batch_precision", precision)
    mlflow.log_metric("batch_recall", recall)
    mlflow.log_metric("batch_f1_score", f1)
    mlflow.end_run()

    return {
        "batch_accuracy": accuracy,
        "batch_precision": precision,
        "batch_recall": recall,
        "batch_f1_score": f1
    }