from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load data
X, y = load_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved in /models/")
