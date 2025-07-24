# NN Regressor with Overclock Mechanism

Welcome. This repo contains a neural network I built from scratch using NumPy. It doesn't use PyTorch, TensorFlow, or any ML library. The key highlight is a mechanism I call **Overclock**, which isn't for increasing the learning rate blindly—it’s about **controlling** it when loss oscillations begin, helping convergence on difficult regressions.

## 📌 Goal

Assume you know nothing about how a neural network works. This project explains everything: forward pass, backpropagation, gradients, dimensions, and training—using vectorized NumPy code for speed and clarity.

---

## 📐 Architecture

This NN is a **fully connected feedforward neural network**. Here’s the setup:

- Input: `X` shape `(m, n)` → `m` samples, `n` features.
- Hidden Layer: 1 layer, custom neurons (can be increased).
- Activation: Swish (`swish(x) = x * sigmoid(x)`)
- Output: Linear activation (for regression).
- Loss: Mean Squared Error (MSE)

---

## ⚙️ Forward Pass

Let:
- `W1` = weights from input to hidden layer → shape `(n, h)`
- `b1` = bias vector for hidden layer → shape `(1, h)`
- `W2` = weights from hidden to output → shape `(h, 1)`
- `b2` = output bias → shape `(1, 1)`

Then, the steps:

```python
Z1 = X @ W1 + b1              # shape: (m, h)
A1 = swish(Z1)                # shape: (m, h)
Z2 = A1 @ W2 + b2             # shape: (m, 1)
Y_pred = Z2                   # shape: (m, 1)
```

Loss:
```python
Loss = np.mean((Y_pred - Y) ** 2)      ## shape: scalar
```

---

## 🔁 Backpropagation: Full Derivation (with Dimensions)

We want to compute gradients for `W1`, `b1`, `W2`, `b2` to do gradient descent.

Let:
- `dZ2 = dLoss/dZ2 = 2 * (Y_pred - Y) / m` → shape `(m, 1)`
- `dW2 = A1.T @ dZ2` → shape `(h, 1)`
- `db2 = np.sum(dZ2, axis=0, keepdims=True)` → shape `(1, 1)`

Now for the hidden layer:

Swish derivative:
```python
sigmoid = lambda x: 1 / (1 + np.exp(-x))
swish_grad = lambda x: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

Then:
```python
dA1 = dZ2 @ W2.T                         # shape: (m, h)
dZ1 = dA1 * swish_grad(Z1)              # shape: (m, h)
dW1 = X.T @ dZ1                         # shape: (n, h)
db1 = np.sum(dZ1, axis=0, keepdims=True)  # shape: (1, h)
```

---

## 🧠 Why Backpropagation Works

The chain rule. That’s it. Backprop computes the partial derivatives of the loss with respect to each parameter, allowing us to take steps in the direction that most reduces the loss.

You update each parameter like this:

```python
W2 -= lr * dW2
b2 -= lr * db2
W1 -= lr * dW1
b1 -= lr * db1
```

The gradients **tell you how much** and **in what direction** each weight should change to minimize the loss. Without them, you'd be guessing blindly.

A partial derivative of a parameter (like a weight or bias) with respect to the loss tells us how sensitive the loss is to changes in that parameter. In other words, it shows how much the loss would change if we tweak that parameter slightly.

If the gradient is positive, it means increasing that parameter will increase the loss — so we should decrease the parameter to reduce the loss.

If the gradient is negative, it means increasing the parameter will decrease the loss — so increasing it is a good move.

But there’s a catch: if you just do parameter += gradient, you'll actually go in the wrong direction, increasing the loss instead of decreasing it. That’s why we flip the sign and do:


parameter -= learning_rate * gradient

Multiplying the gradient by -1 makes sure we move in the direction that minimizes the loss.

---

## 🌀 Overclock Mechanism

This isn't a standard LR scheduler. Here's the idea:

1. Use an aggressive learning rate (`lr = 0.1`) with no decay.
2. Train one model fully like this.
3. Then, take another model initialized with the **same random seed/weights**.
4. Watch the loss. When it **starts oscillating**, activate Overclock:
   - Set `lr += overclock_amount` (can be negative).
   - This "Overclock" slows the training at just the right moment.

It’s like **manually controlled learning rate decay** based on observing the training curve.

This avoids early plateau or divergence that a static scheduler can’t catch.

---

## 📚 Why Use Learning Rates?

Because gradients alone aren’t enough. You need a scalar multiplier (`lr`) to control the **step size**. Too small = slow. Too large = divergence. Overclock gives you the control to time it just right.

---

## ✅ Coming Soon

I'll link to my other repo that uses **pure Python**, without NumPy—made for educational purposes to show how neural networks really work, one scalar at a time.

---

## 🔓 Final Thoughts

This project helped me understand backpropagation, gradients, and the importance of inspecting loss trends manually. The Overclock mechanism wasn’t born from theory—it came from **observation** and **experimentation**.



Stay tuned for more.
