# 🔁 Backpropagation on Feedforward Neural Network

This project demonstrates training a simple multilayer feedforward neural network using the **Backpropagation algorithm** on a single training tuple: **(X1 = 1, X2 = 1, Target = 0)**.

---

## 🧠 Network Architecture

- **Input Layer:** 2 neurons (X1, X2)
- **Hidden Layer:** 2 neurons (Node 3, Node 4)
- **Output Layer:** 1 neuron (Node 5)
- **Activation Function:** Sigmoid
- **Training Method:** Backpropagation
- **Learning Rate:** 0.5

---

## 📐 Initial Weights and Biases

| Connection | Value  |
|------------|--------|
| W13        | 0.5    |
| W23        | -0.3   |
| b3         | 0.6    |
| W14        | 0.2    |
| W24        | 0.5    |
| b4         | -0.4   |
| W35        | 0.1    |
| W45        | 0.3    |
| b5         | 0.8    |

---

## 🧪 Training Details

- **Input:** (X1 = 1, X2 = 1)
- **Target Output:** 0
- **Training Epochs:** 1 (Single forward and backward pass)
- **Activation Function:**
  \
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{and} \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))
  \]

---

## 💻 Python Code (with Explanations)

```python
import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid (for backpropagation)
def sigmoid_derivative(a):
    return a * (1 - a)

# Inputs and target
X1, X2 = 1, 1
target = 0
lr = 0.5  # learning rate

# Initial weights and biases
W13, W23, b3 = 0.5, -0.3, 0.6  # connections to hidden neuron 3
W14, W24, b4 = 0.2, 0.5, -0.4  # connections to hidden neuron 4
W35, W45, b5 = 0.1, 0.3, 0.8   # connections to output neuron 5

# --------- Forward Pass ---------

# Hidden neuron 3
z3 = X1 * W13 + X2 * W23 + b3
a3 = sigmoid(z3)

# Hidden neuron 4
z4 = X1 * W14 + X2 * W24 + b4
a4 = sigmoid(z4)

# Output neuron 5
z5 = a3 * W35 + a4 * W45 + b5
a5 = sigmoid(z5)

print("Forward Pass Output:")
print(f"a3 = {a3:.5f}, a4 = {a4:.5f}, a5 (output) = {a5:.5f}\n")

# --------- Backward Pass ---------

# Error at output
error = a5 - target

# Output neuron delta
delta5 = error * sigmoid_derivative(a5)

# Hidden neuron deltas
delta3 = delta5 * W35 * sigmoid_derivative(a3)
delta4 = delta5 * W45 * sigmoid_derivative(a4)

# --------- Update Weights and Biases ---------

# Output layer
W35 -= lr * delta5 * a3
W45 -= lr * delta5 * a4
b5  -= lr * delta5

# Hidden layer
W13 -= lr * delta3 * X1
W23 -= lr * delta3 * X2
b3  -= lr * delta3

W14 -= lr * delta4 * X1
W24 -= lr * delta4 * X2
b4  -= lr * delta4

# Final updated values
print("Updated Weights and Biases after 1 training example:")
print(f"W13 = {W13:.5f}, W23 = {W23:.5f}, b3 = {b3:.5f}")
print(f"W14 = {W14:.5f}, W24 = {W24:.5f}, b4 = {b4:.5f}")
print(f"W35 = {W35:.5f}, W45 = {W45:.5f}, b5 = {b5:.5f}")
print(f"Loss = {0.5 * error**2:.5f}")
```

---

## 📊 Output of This Script

Forward Pass Output:
a3 = 0.68997, a4 = 0.57444, a5 (output) = 0.73911

Updated Weights and Biases after 1 training example:
W13 = 0.49848, W23 = -0.30152, b3 = 0.59848
W14 = 0.19477, W24 = 0.49477, b4 = -0.40523
W35 = 0.05083, W45 = 0.25907, b5 = 0.72874
Loss = 0.27314
---

## 🚀 How to Run the Code

1. Save the code to a Python file, e.g., `backpropagation_train.py`
2. Run it:
   ```bash
   python backpropagation_train.py
   ```

---

## 📚 Result

This script performs a full forward and backward pass using the backpropagation algorithm and shows how weights are updated based on the error.
