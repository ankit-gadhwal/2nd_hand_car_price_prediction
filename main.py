from fastapi import FastAPI
from tensorflow.keras.models import load_model
import pandas as pd
import uvicorn

# ----------------------------------------
# Initialize FastAPI App
# ----------------------------------------
app = FastAPI()

# ----------------------------------------
# Load Trained Model
# ----------------------------------------
model = load_model("car_price_model.h5")

# ----------------------------------------
# Home Route
# ----------------------------------------
@app.get("/")
def home():

    return {
        "message": "2nd Hand Car Price Prediction API Running"
    }


# ----------------------------------------
# Prediction Route
# ----------------------------------------
@app.get("/predict")
def predict(

    on_road_old: float,
    on_road_now: float,
    years: float,
    km: float,
    rating: float,
    condition: float,
    economy: float,
    top_speed: float,
    hp: float,
    torque: float

):

    try:

        # ----------------------------------------
        # Feature Engineering
        # ----------------------------------------
        depreciation = (
            on_road_old -
            on_road_now
        )

        km_per_year = (
            km / (years + 1)
        )

        performance = (
            hp + torque
        )

        value_score = (
            economy * rating
        )

        # ----------------------------------------
        # Create Input DataFrame
        # ----------------------------------------
        input_data = pd.DataFrame([{

            "on road old": on_road_old,
            "on road now": on_road_now,
            "years": years,
            "km": km,
            "rating": rating,
            "condition": condition,
            "economy": economy,
            "top speed": top_speed,
            "hp": hp,
            "torque": torque,

            # Engineered Features
            "depreciation": depreciation,
            "km_per_year": km_per_year,
            "performance": performance,
            "value_score": value_score

        }])

        # ----------------------------------------
        # Prediction
        # ----------------------------------------
        prediction = model.predict(
            input_data
        )[0][0]

        return {

            "Predicted Car Price": round(
                float(prediction),
                2
            )
        }

    except Exception as e:

        return {
            "Error": str(e)
        }


# ----------------------------------------
# Run Server
# ----------------------------------------
if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )