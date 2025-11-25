import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dummy training data (just for demo)
X = np.array([
    [20, 0],
    [25, 5],
    [30, 10],
    [40, 15],
    [50, 20],
    [60, 30],
])

y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

with open("lung_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as lung_risk_model.pkl!")
