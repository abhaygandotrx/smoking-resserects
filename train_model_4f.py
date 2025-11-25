import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# ----------- Generate synthetic training data --------------
np.random.seed(42)
data_size = 400

age = np.random.randint(18, 80, data_size)
smoking_years = np.random.randint(0, 40, data_size)
cough_freq = np.random.randint(0, 50, data_size)
smoke_detected = np.random.randint(0, 2, data_size)

# Risk rule (fake but realistic)
risk = (age > 50) | (smoking_years > 15) | (cough_freq > 20) | (smoke_detected == 1)
risk = risk.astype(int)

df = pd.DataFrame({
    "age": age,
    "smoking_years": smoking_years,
    "cough_freq": cough_freq,
    "smoke_detected": smoke_detected,
    "risk": risk
})

# ----------- Train model --------------
X = df[["age", "smoking_years", "cough_freq", "smoke_detected"]]
y = df["risk"]

model = LogisticRegression()
model.fit(X, y)

# ----------- Save model --------------
with open("lung_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("4-feature model trained and saved as lung_risk_model.pkl")
