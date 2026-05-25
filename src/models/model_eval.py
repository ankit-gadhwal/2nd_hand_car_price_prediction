import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
def data_load(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error occured in data loading {e}")
    
def data_prepare(df):
    try:
        X_test = df.drop(columns = "current price")
        y_test = df["current price"]
        return X_test,y_test
    except Exception as e:
        raise Exception(f"error occured in data preparation {e}")
    
def save_metrics(metrics,filepath):
    try:
        with open(filepath,"w") as file:
            json.dump(metrics,file,indent=4)
            
    except Exception as e:
        raise Exception(f"Error saving metrics: {e}")
    
def main():
    try:
        processed_path = os.path.join("data","processed")
        reports_path = "reports"
        os.makedirs(
            reports_path,
            exist_ok=True
        )

        # load model
        model = load_model("car_price_model.h5")
        
        # load test data
        test_df = data_load(os.path.join(processed_path,"test_processed.csv"))

        # prepare data
        X_test,y_test = data_prepare(test_df)

        # Evaluate model
        loss,rmse = model.evaluate(
            X_test.values,
            y_test.values,
            verbose=0
        )

        # predictions
        y_pred = model.predict(X_test.values)
        y_pred = y_pred.flatten()

        # Metrics
        metrics = {
            "mae": float(mean_absolute_error(y_test,y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test,y_pred))),
            "r2_score":float(r2_score(y_test,y_pred))
        }

        # save metrics
        save_metrics(metrics,os.path.join(reports_path,"metrics.json"))
        
    except Exception as e:
        raise Exception(f"Error occured: {e}")   
    
if __name__ == "__main__":
    main()