# README: Perceptron Logic Gate Classifier

## Overview

This is a simple implementation of a **Perceptron** from scratch using NumPy and Matplotlib in Python. The code demonstrates how a single-layer perceptron can learn to classify **AND gate** outputs. It also plots **true vs predicted outputs** during each training epoch to visually show the learning progress.

## What is a Perceptron?

A Perceptron is the most basic type of neural network, used for **binary classification**. It makes decisions by calculating a weighted sum of input features, adding a bias, and passing the result through an **activation function** (in this case, a step function).

## Code Breakdown

- **Input (`X`)**: 4 combinations for 2-input gates: `[[0,0], [0,1], [1,0], [1,1]]`
- **Labels (`y`)**: The output of the AND gate for each input.
- **Weights (`w`)**: Randomly initialized, updated through learning.
- **Activation Function**: Step function returns `1 if z > 0`, else `0`.
- **Training**: 20 epochs using simple Perceptron learning rule.
- **Visualization**: Plots **True vs Predicted** outputs for every epoch.

**Simple Preceptron:**
A perceptron is a **binary linear classifier** that:
- Takes multiple input features.
- Computes a weighted sum of the inputs.
- Applies an activation function (e.g., step function) to produce an output (typically `0` or `1`).

## Mathematical Model
### Inputs and Weights
Given:
- Input vector: **X** = [x₁, x₂, ..., xₙ]
- Weight vector: **W** = [w₁, w₂, ..., wₙ]
- Bias: `b`
### Weighted Sum (Pre-Activation)
The perceptron computes: z = sum(i=1)^n = (W_i * X_i)+ b_i

### Activation Function
A **step function** is applied to `z`: 
f(z) = Activation(z) # A = Relu, Sigmoid, Tanh or LeakyRelu

## Learning Rule
The perceptron updates its weights iteratively using the **Perceptron Learning Algorithm**:
1. **Initialize** weights (`w`) and bias (`b`) to small random values or `0`.
2. **For each training sample** (xᵢ, yᵢ):
   - Compute predicted output: y_pred = Activation(f(z))
   - Update Weight if y_pred != y_true
   - Weight updation : w_i = w_i + Learning_rate * (y_true - y_pred) * x_i
   - Bias Updation : b + Learning_rate * (y_true - y_pred)
   - `α` = learning rate (usually `0 < α ≤ 1`).

## Limitations
- **Linearly Separable Data**: Only works if data can be divided by a hyperplane.
- **XOR Problem**: Cannot solve non-linear problems (e.g., XOR), leading to the development of multi-layer perceptrons (MLPs).

****Multiple Layer Preceptron (MLP):****
1. Forward Propagation:
   
   A. Weighted Input
      It’s the process of passing inputs through layers of the neural network to get the output (prediction).
    
       Mathematical Formula - z = W.x + b
       W = Weights corresponding to that neuron
       B =  Bias Corresponding to that nueron
       X = Input for the neuron
   B. Activation Function
      Each neuron computes a weighted sum of inputs plus bias and applies an activation function to introduce non-linearity.

      A = f(z)
      A = Relu, Sigmoid, Tanh or LeakyRelu
   
2. Backward Propagation:
   A. It calculates the error between prediction and actual output.
   B. Gradients (slopes) are calculated using the chain rule.
   C. Weights are updated in the direction that reduces the error (gradient descent).

   Mathematical Formula - dloss/dweight = dloss/dactivation X dactivation/dweightedinput x dweightedinput/dweight
   
   Goal: Minimize the loss by updating W and B.
   


## How to Run

Make sure you have Python installed along with NumPy and Matplotlib.

```bash
pip install numpy matplotlib
```

Then run the Python script:

```bash
python perceptron_and_gate.py
```

## Output

At each epoch, you'll see:

- Printed **True and Predicted values**
- A plot showing the prediction progress for the 4 input samples

## Truth Tables for Logic Gates

| Input A | Input B | AND | OR  | NAND | NOR | XOR |
|--------:|--------:|----:|----:|-----:|----:|----:|
|   0     |    0    |  0  |  0  |  1   |  1  |  0  |
|   0     |    1    |  0  |  1  |  1   |  0  |  1  |
|   1     |    0    |  0  |  1  |  1   |  0  |  1  |
|   1     |    1    |  1  |  1  |  0   |  0  |  0  |

## Note

- The **Perceptron** can **only solve linearly separable problems**. So, it works for AND, OR, NAND, NOR — **but not XOR**, which is not linearly separable.
- To solve XOR, you need a **multi-layer perceptron (MLP)**.

## File Structure

```
perceptron_and_gate.py     # Python script with perceptron logic
README.md                  # This file
```

## Extensions (Ideas for You)

- Modify the labels `y` to try **OR**, **NAND**, or **NOR** gates.
- Plot the **decision boundary** (for visual understanding).
- Implement a **Multi-layer Perceptron (MLP)** to solve **XOR** gate.
- Add accuracy tracking or loss function graphs.
