import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-8  # avoid log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

# Feedforward
def feedforward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Backpropagation
def backpropagation(X, y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]  # Number of samples

    # Output layer error
    dZ2 = (A2 - y) * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer error
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# Load and prepare the dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the target
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Normalize inputs
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Network architecture
input_size = X.shape[1]  # 4 features
hidden_size = 5          # Hidden layer size (tune as needed)
output_size = y_encoded.shape[1]  # 3 classes

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Training loop
epochs = 500
learning_rate = 0.1

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Forward pass
    Z1, A1, Z2, A2 = feedforward(X_train_shuffled, W1, b1, W2, b2)

    # Backward pass
    dW1, db1, dW2, db2 = backpropagation(X_train_shuffled, y_train_shuffled, Z1, A1, Z2, A2, W2)

    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Compute loss
    loss = np.mean(np.square(A2 - y_train_shuffled))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Evaluate on test set
_, _, _, A2_test = feedforward(X_test, W1, b1, W2, b2)
predictions = np.argmax(A2_test, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == actual)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
