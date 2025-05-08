Neural Network with Backpropagation

This project implements a basic feedforward neural network **from scratch using NumPy**, including the **backpropagation algorithm**, to classify the **Iris dataset**. It is built for **COMS4030A: Adaptive Computation and Machine Learning (ACML)** Lab 3 exercise.

## 🔧 Features

- Single hidden layer neural network
- Sigmoid activation functions (hidden + output)
- Sum-of-squares loss function
- One-hot encoding of target labels
- Random shuffling of data before each epoch
- Manual weight and bias updates using gradients
- Train/test evaluation with accuracy metric

---

## 📁 Project Structure
├── main.py # Main script to train and test the neural network\\
├── README.md # This file\\
├── .gitignore # To ignore virtual environment and other unwanted files\\
└── requirements.txt # List of Python dependencies\\

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/backpropagation-Algorithm.git
cd backpropagation-Algorithm

##Setting up a Virtual Environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate


## Install Dependencies
pip install -r requirements.txt

#To regenerate requirements.txt manually:
pip freeze > requirements.txt

## Or manually install required packages:
pip install numpy scikit-learn


##Run the Code
Python backprop_algo.py


