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
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

# Backpropagation with softmax + cross-entropy
def backpropagation(X, y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]

    dZ2 = (A2 - y) / m  # cross-entropy derivative with softmax
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# Load and prepare dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Normalize inputs
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model architecture
input_size = X.shape[1]
hidden_size = 10  # increased hidden layer size
output_size = y_encoded.shape[1]

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, output_size))

# Training
epochs = 500
learning_rate = 0.05

for epoch in range(epochs):
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Forward
    Z1, A1, Z2, A2 = feedforward(X_train_shuffled, W1, b1, W2, b2)

    # Backward
    dW1, db1, dW2, db2 = backpropagation(X_train_shuffled, y_train_shuffled, Z1, A1, Z2, A2, W2)

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Monitor loss and accuracy
    if epoch % 10 == 0 or epoch == epochs - 1:
        loss = cross_entropy_loss(A2, y_train_shuffled)
        predictions = np.argmax(A2, axis=1)
        targets = np.argmax(y_train_shuffled, axis=1)
        accuracy = np.mean(predictions == targets)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

# Evaluate on test set
_, _, _, A2_test = feedforward(X_test, W1, b1, W2, b2)
test_predictions = np.argmax(A2_test, axis=1)
test_actual = np.argmax(y_test, axis=1)
test_accuracy = np.mean(test_predictions == test_actual)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
