# XOR Problem - 3-Layer Neural Network using NumPy

This project implements a simple 3-layer feedforward neural network from scratch using **NumPy** to solve the **XOR binary classification problem**.

---

## Problem Description

- **Dataset**: XOR gate truth table:

| Input (X1, X2) | Output (Y) |
| -------------- | ---------- |
| [0, 0]         | 0          |
| [1, 0]         | 1          |
| [1, 1]         | 0          |
| [0, 1]         | 1          |

- **Goal**: Train a neural network to classify the XOR outputs correctly using forward propagation, backward propagation, and gradient descent.

---

## Network Architecture

| Layer              | Details                                                  |
| ------------------ | -------------------------------------------------------- |
| **Input Layer**    | 2 neurons (X1, X2)                                       |
| **Hidden Layer 1** | 4 neurons, ReLU activation                               |
| **Hidden Layer 2** | 4 neurons, ReLU activation                               |
| **Output Layer**   | 1 neuron, Sigmoid activation (for binary classification) |

---

## Components Explained

### 1. **Forward Propagation**

- Calculates activations at each layer using:
  - Linear step: `Z = X·W + b`
  - Activation functions:
    - **ReLU** for hidden layers
    - **Sigmoid** for output layer

### 2. **Loss Calculation**

- **Binary Cross-Entropy Loss**:

$$
Loss = Cross Entrophy Loss
$$

### 3. **Backward Propagation**

- Uses **gradients of the loss** to compute updates for each layer:
  - `dZ3 = A3 - Y` (from sigmoid and BCE derivative simplification)
  - Gradients are propagated backward through layers.

### 4. **Weight Update**

- **Gradient Descent** is used to update weights and biases:

$$
W = W - Learning_rate * dW
$$

---

##  Hyperparameters

| Hyperparameter | Value                    |
| -------------- | ------------------------ |
| Learning Rate  | 0.1                      |
| Epochs         | 10,000                   |
| Batch Size     | 4 (entire dataset)       |
| Loss Function  | Binary Cross-Entropy     |
| Optimizer      | Vanilla Gradient Descent |

---

## Expected Output

- The **loss** decreases progressively and the model **predicts the XOR pattern correctly** after training.
- Example Output:

```
Epoch 0 Loss: 0.8097
Epoch 1000 Loss: 0.0205
...
Epoch 9000 Loss: 0.0057
Predictions:
 [[0]
 [1]
 [0]
 [1]]
```

---

You can modify it to plot **loss over epochs** like:

```python
losses = []
for epoch in range(epochs):
    ...
    losses.append(loss)

plt.plot(losses)
plt.title("Loss Curve over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

---

## How to Run

1. Install Python 3.x
2. Install NumPy and Matplotlib:

```bash
pip install numpy matplotlib
```

3. Run the Python script:

```bash
python xor_neural_network.py
```

---

## Learning Notes

- ReLU activation keeps the hidden layers non-linear.
- Sigmoid activation converts final output into a probability.
- `A3 - Y` term comes from the derivative of BCE loss with sigmoid — simplifying backpropagation.
- Using a simple MLP, you can solve non-linearly separable problems like XOR!

---

