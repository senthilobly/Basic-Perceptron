# README: Perceptron

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
   A. It calculates the error(Loss Function) between prediction and actual output.
   B. Gradients (slopes) are calculated using the chain rule.
   C. Weights are updated in the direction that reduces the error (gradient descent).

   Mathematical Formula - dloss/dweight = dloss/dactivation X dactivation/dweightedinput x dweightedinput/dweight
   
   Goal: Minimize the loss by updating W and B.

3. Loss Calculation:
   It measures how “wrong” the model’s predictions are.
   | Type              | Use-Case       | Formula                                                              |
   | ----------------- | -------------- | ---------------------------------------------------------------------|
   | **MSE**           | Regression     | `MSE = (sum of (actual - predicted)^2) / number of data`             |
   | **Cross-Entropy** | Classification | `Cross-Entropy = - (y * log(y_pred) + (1 - y) * log(1 - y_pred))`    |

   Interpretation:
   Lower loss = better predictions.
   Optimizer’s goal = minimize loss.

   Example:
   If prediction = 0.9, true label = 1:
   Cross-Entropy Loss = −log(0.9)=0.105

## Neural Network Training Stabilizers

### Neural Network Training Stabilizers

| Technique          | Purpose                                                                 | Key Effects                                                                 |
|--------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Regularization** (L1/L2) | Prevents overfitting by penalizing large weights. | • L1: Creates sparse weights (some → 0) <br> • L2: Keeps weights small but non-zero |
| **Dropout**        | Randomly disables neurons during training.       | • Reduces overfitting <br> • Prevents neuron co-dependency                   |
| **Batch Norm**     | Normalizes layer outputs to stabilize training.  | • Faster convergence <br> • Reduces sensitivity to initialization           |
| **Optimization** (Adam/SGD) | Efficiently updates model weights.       | • Avoids local minima <br> • Adapts learning rates dynamically              |

**Why They Matter**:  
These techniques improve generalization, speed up training, and prevent overfitting in deep learning models.

**Regularization:**
   Regularization controls the complexity of a neural network by adding a penalty to large weights in the loss function.        It prevents overfitting and improves generalization to unseen data.

   Types include L1 (Lasso) and L2 (Ridge) regularization.

   ### Regularization Techniques Comparison

| Type                     | Concept                          | Math Formula                     | Usage/When to Use                          | Effect                                      |
|--------------------------|----------------------------------|----------------------------------|--------------------------------------------|---------------------------------------------|
| **L1 (Lasso)**          | Penalizes absolute weights       | `Loss + λ * Σ\|W\|`             | Sparse models / feature selection          | Zeroes out unimportant weights             |
| **L2 (Ridge)**         | Penalizes squared weights        | `Loss + λ * ΣW²`                | General overfitting prevention             | Smoothly shrinks all weights               |
| **Elastic Net**        | Combines L1 + L2                 | `Loss + λ1*Σ\|W\| + λ2*ΣW²`     | Need both feature selection and stability  | Balances sparsity and weight shrinkage     |

#### Key:
- **λ (lambda)**: Controls regularization strength
- **Σ\|W\|**: Sum of absolute weights (L1)
- **ΣW²**: Sum of squared weights (L2)

**Dropout:**
Dropout randomly disables neurons during training to prevent over-reliance on specific neurons. It forces the network to learn redundant and robust features. This reduces overfitting, especially in large neural networks.

### Dropout Regularization

| Concept        | Math Formula (Simplified)       | Usage/When to Use                          | Effect on Model                              | Typical Values          |
|----------------|----------------------------------|--------------------------------------------|----------------------------------------------|-------------------------|
| **Dropout**   | `A_drop = A * mask / (1 - p)`  | Large networks (CNNs/MLPs) prone to overfitting | Adds robustness, prevents neuron co-dependence | `p = 0.1` to `0.5` (0.5 common) |
| **Backward Pass** | `dZ = dZ * mask / (1 - p)` | Applied during backpropagation | Maintains gradient consistency | Same as forward pass |

#### Key:
- `A`: Layer activations
- `mask`: Binary matrix (Bernoulli distribution)
- `p`: Dropout probability (e.g., 0.2 = 20% neurons dropped)
- `1/(1-p)`: Scaling factor (inverted during training)

**Batch Normalization:**
Batch Normalization standardizes the activations within each mini-batch. It reduces internal covariate shift, leading to faster and more stable training. It can also allow for higher learning rates and less overfitting.

### Batch Normalization

| Concept               | Math Formula (Simplified)                     | Usage/When to Use                          | Effect on Model                              | Notes                                      |
|-----------------------|-----------------------------------------------|--------------------------------------------|----------------------------------------------|--------------------------------------------|
| **Batch Norm**       | `x_norm = (x - μ) / √(σ² + ε)`<br>`x_out = γ * x_norm + β` | Deep networks, unstable training | Stabilizes training, faster convergence | Used after layers but before activation |
| **Inference Mode**   | Uses running averages: `μ_running`, `σ²_running` | Production/prediction phase | Consistent behavior on new data | Disables batch-dependent calculations |

#### Key:
- `μ/σ²`: Batch mean/variance
- `ε`: Small constant (e.g., 1e-5)
- `γ/β`: Learnable scale/shift parameters
- Running stats: Updated during training with momentum

#### Typical Values:
- `ε = 1e-5`
- Momentum for running stats: `0.99`

**Optimizers:**
Optimizers adjust weights using gradients to minimize loss efficiently. They determine how fast and stable the learning progresses. Popular optimizers include SGD, Momentum, RMSProp, Adam, with Adam being the most widely used.

### Optimization Algorithms

| Optimizer  | Math Formula (Simplified) | Usage/When to Use | Effect on Learning | Key Parameters |
|------------|--------------------------|------------------|-------------------|----------------|
| **SGD**    | `W = W - α*∇L` | Small datasets/models | Baseline, stable but slow | Learning rate (α) |
| **Momentum** | `v = β*v + (1-β)*∇L`<br>`W = W - α*v` | Zig-zag gradients | Faster convergence | α, β (~0.9) |
| **RMSProp** | `v = β*v + (1-β)*∇L²`<br>`W = W - α*∇L/(√v + ε)` | RNNs/unstable training | Stabilizes updates | α, β, ε (1e-8) |
| **Adam**   | `m = β1*m + (1-β1)*g`<br>`v = β2*v + (1-β2)*g²`<br>`W = W - α*m/(√v + ε)` | Most deep learning tasks | Fast convergence | α, β1 (0.9), β2 (0.999), ε |
| **AdamW**  | Adam + weight decay | Regularized models | Better generalization | Same as Adam |

#### Key:
- `W`: Weights
- `α`: Learning rate
- `β/β1/β2`: Momentum terms
- `ε`: Small constant (~1e-8)
- `g/∇L`: Gradient

📝 Summary Cheat Notes:
Topic	Use Case
Regularization	- Controls weight size to avoid overfitting
Dropout	- Randomly drops neurons to prevent co-dependency
Batch Norm	- Normalizes activations to stabilize training
Optimizers	- Smart weight updates for faster, stable convergence


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
