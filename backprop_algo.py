import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data  # shape: (150, 4)
y = iris.target.reshape(-1, 1)  # shape: (150, 1)

# One-hot encode the labels: (150, 1) => (150, 3)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Normalize inputs
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

input_size = 4
hidden_size = 5  # you can tweak this
output_size = 3  # 3 classes in Iris dataset

# Initialize weights and biases with small random values
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

