# Import Libraries
from utils import process_new
import joblib
import os
import numpy as np
from fastapi import FastAPI


# Load the Model
MODEL_PATH = os.path.join(os.getcwd(),'Model_XGBoost.pkl')
model =joblib.load(MODEL_PATH)

# Initialize an app
app = FastAPI()

@app.get('/root')

async def root(total_volume: float,plus_4046: float,plus_4225: float,plus_4770: float,total_bags: float,
         small_bags:float,large_bags:float,xlarge_bags:float,type:str,year:int,region:str):
    
    # Concatenate
    new_data= np.array([total_volume,plus_4046,plus_4225,plus_4770,total_bags,
         small_bags,large_bags,xlarge_bags,type,year,region])
    
    # Call the function from utils.py
    X_processed= process_new(X_new=new_data)

    # Model Prediction
    y_pred = model.predict(X_processed)[0]



    return {f'Avocado price is {y_pred}'}
