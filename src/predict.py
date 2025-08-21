import joblib
import pandas as pd

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/svm_model.pkl")

feature_names = ["sepal length (cm)", "sepal width (cm)", 
                 "petal length (cm)", "petal width (cm)"]

def predict_iris(features):
    # Convert to DataFrame with correct feature names
    df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(df)
    prediction = model.predict(features_scaled)[0]
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return species_map[prediction]

if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]
    print("Predicted species:", predict_iris(sample))
