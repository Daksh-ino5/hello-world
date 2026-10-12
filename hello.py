Experiment 1: Linear Regression and Regularization

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Add bias column
Xb = np.c_[np.ones(len(X_train)), X_train]
X_test_b = np.c_[np.ones(len(X_test)), X_test]

# 1. Normal Equation
theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y_train
pred = X_test_b @ theta
print("Normal Equation MSE:", mean_squared_error(y_test, pred))

# 2. Gradient Descent
theta = np.zeros(Xb.shape[1])
lr = 0.01
for i in range(1000):
    error = Xb @ theta - y_train
    theta -= lr * (2 / len(y_train)) * (Xb.T @ error)

pred = X_test_b @ theta
print("Gradient Descent MSE:", mean_squared_error(y_test, pred))

# 3. Linear, Ridge and Lasso
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "MSE:", mean_squared_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))


****--------------------------------------------------------------------------------------------****

Experiment 2: Logistic Regression and Perceptron

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression using Gradient Descent
w = np.zeros(X_train.shape[1])
b = 0
for _ in range(1000):
    p = 1 / (1 + np.exp(-np.clip(X_train @ w + b, -500, 500)))
    e = p - y_train
    w -= 0.01 * (X_train.T @ e) / len(y_train)
    b -= 0.01 * np.mean(e)

pred1 = (1 / (1 + np.exp(-np.clip(X_test @ w + b, -500, 500))) >= 0.5).astype(int)

# Perceptron
w = np.zeros(X_train.shape[1])
b = 0
for _ in range(100):
    for x, y1 in zip(X_train, y_train):
        t = 1 if x @ w + b >= 0 else 0
        if t != y1:
            w += 0.01 * (2 * y1 - 1) * x
            b += 0.01 * (2 * y1 - 1)

pred2 = (X_test @ w + b >= 0).astype(int)

# Accuracy
print("Logistic Regression:", accuracy_score(y_test, pred1))
print("Perceptron:", accuracy_score(y_test, pred2))