import joblib
from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load saved model & scaler
model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Scale test data
X_test = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
