# Overclock Neural Network (OCNN)

The **Overclock Neural Network** is a custom-built neural network written in NumPy, designed to explore learning rate dynamics, fast convergence strategies, and deeper mathematical understanding of backpropagation. It introduces an "Overclock" mechanism — an intuitive way to control the speed of learning — while keeping full transparency into the inner workings of training.

This README explains **why** each part of the neural network exists, and **how** it all fits together.

---

## 🚀 Why Use Neural Networks?

Neural Networks can **approximate complex, non-linear functions**. They're inspired by how neurons fire in the brain, but more importantly, they let us learn patterns in data — be it numbers, images, sounds, or actions.

You could train a neural network to learn anything from predicting fruits to detecting objects in real-time. But to understand them deeply, you have to know *how* each component affects the outcome — mathematically and intuitively.

---

## 🧠 Why Do We Need Activation Functions?

### The Problem:
Without activation functions, a neural network becomes just a stack of linear transformations — matrix multiplications and bias additions. Mathematically:

```
y = W2 · (W1 · x + b1) + b2
```

This whole thing simplifies to a single linear transformation:

```
y = W · x + b
```

It can't learn *non-linear* patterns like `sin(x)`, XOR, or digit shapes.

### The Solution:
Activation functions like `ReLU`, `Swish`, or `Sigmoid` introduce **non-linearity** into the network. That means your model can now bend the space and curve the decision boundary, rather than just drawing straight lines.

**Swish activation**, which you use, is defined as:

```
swish(x) = x * sigmoid(x)
```

Why swish?
- It's **smooth** and **non-monotonic**
- Allows better gradient flow than ReLU in some cases
- Encourages small negative values instead of zeroing them out

---

## 🎯 How Does a Neural Network Learn?

Neural networks learn by adjusting their parameters (weights and biases) to reduce the error between predicted output and actual target. This process is called **training**, and it's powered by **gradient descent**.

---

## ⚙️ Why Use Gradient Descent?

The neural network has millions of possible parameter combinations. The goal is to find one that gives the **least error** (or loss).

We treat the **loss function** like a landscape and try to move downhill towards the lowest point. But how do we know where to step?

We use **gradients** — partial derivatives of the loss with respect to each parameter — to guide us.

A gradient tells you:
- *How sensitive* the loss is to a change in that parameter
- *Which direction* the parameter should move in to reduce the loss

If the gradient is positive → increasing the parameter increases the loss  
If the gradient is negative → increasing the parameter decreases the loss

So we go **against** the gradient:

```
parameter = parameter - learning_rate * gradient
```

This is the essence of **gradient descent**.

---

## ⚡ Why Use a Learning Rate?

Learning rate is a **scaling factor** that decides how big a step we take during training.

- If it's too small → the model learns too slowly
- If it's too large → the model overshoots and might never converge

The Overclock NN uses a dynamic learning rate idea — it lets you tweak this "overclock" to balance **speed** and **stability**.

**Overclocking** here means pushing the learning rate higher than usual while using checks like gradient clipping or decay to avoid divergence. It's like putting your model in turbo mode — but with control.

---

## 🧮 How Does Backpropagation Work?

Backpropagation is a method for computing the **gradients** of the loss with respect to every parameter in the network.

Here’s the process:

1. **Forward Pass**: Compute the prediction by passing inputs through the network.
2. **Loss Calculation**: Compare prediction with true label using a loss function like MSE or Cross-Entropy.
3. **Backward Pass**:
    - Apply the chain rule to compute `dL/dW` and `dL/db` for each layer.
    - These gradients tell how much each weight/bias contributed to the error.
4. **Parameter Update**: Apply gradient descent to update parameters.

Mathematically:

```
Z1 = W1 · X + b1
A1 = swish(Z1)

Z2 = W2 · A1 + b2
A2 = prediction

Loss = MSE(A2, Y)

dLoss/dW2 = (A2 - Y) · A1.T
dLoss/dW1 = ((W2.T · (A2 - Y)) ⊙ swish'(Z1)) · X.T
```

Where ⊙ is element-wise multiplication, and `swish'(x)` is the derivative of Swish.

---

## 🧪 Why Use Mini-Batch (Matrix-Based) Training?

Instead of feeding one sample at a time, you can process a whole **batch** of inputs at once using matrices. This speeds up computation (especially on GPUs) and improves gradient estimation.

Matrix shapes:

- `X`: shape `(features, batch_size)`
- `W`: shape `(neurons, features)`
- `Z`: shape `(neurons, batch_size)`
- `A`: shape `(neurons, batch_size)`

Matrix-based training = fast, efficient, and works well with libraries like NumPy.

---

## 💡 What Makes Overclock NN Special?

- **Swish activation**: smooth and expressive
- **Overclock mechanism**: tweak learning rate to speed up training
- **Modular code**: easy to plug in new activations, optimizers, loss functions
- **Pure NumPy**: clean, transparent logic without hidden black boxes

---

## 📚 How to Use

```python
from ocnn import OverclockNN

model = OverclockNN(input_size=4, hidden_size=10, output_size=3, activation='swish')

model.train(X_train, Y_train, epochs=3000, batch_size=16, overclock_factor=1.5, learning_rate=0.1)

predictions = model.predict(X_test)
```

---

## 🛠️ Roadmap

- Add learning rate decay
- Add other activations (Leaky ReLU, Tanh)
- Add classification metrics (accuracy, F1)
- Save and load weights

---

## 📎 Summary of Core Ideas

- **Why gradient descent?** To find the direction that reduces error
- **Why learning rates?** To control the step size in training
- **Why activations?** To model non-linear patterns
- **Why matrices?** To scale training efficiently
- **Why swish?** For smoother gradients and richer learning
- **Why overclock?** To learn faster when it works, and back off when it doesn’t

---

## 🔗 Related Projects

- [Pure Python NN](./pure_python_nn/README.md) — a from-scratch neural network without NumPy

---

## 🧠 Final Thought

Neural networks aren’t just code — they’re **mathematical ideas brought to life**. This project aims to make every step visible and intuitive, while still being fast and scalable.

Use it. Break it. Understand it.
```
