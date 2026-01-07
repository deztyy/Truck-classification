import mlflow
import numpy as np

# 1. Define the Model URI (Name/Version)
# Options: 'models:/ModelName/1' or 'models:/ModelName/Production'
model_uri = "models:/Truck_classification_Model/Production"

# 2. Load the model
model = mlflow.pyfunc.load_model(model_uri)

print("Load Model successfully")
print(f"The model is temporarily downloaded at: {model.metadata.get_model_info().model_uri}")


