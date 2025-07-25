# Custom Neural Network Architecture – README

This project is a from-scratch neural network built with raw Python, no external ML libraries used. It’s designed to teach and showcase the **inner logic behind neural networks**, with special attention to how and why certain components exist in the architecture.

## Why Use Activation Functions?

Activation functions introduce **non-linearity** into the neural network. 

Without them, your model—no matter how many layers deep—is just doing a series of matrix multiplications, which essentially turns it into a **big linear model**. It won’t be able to learn complex patterns like XOR, or classify anything beyond linearly separable data.

By adding activations (like ReLU, Sigmoid, Tanh, etc.), you're giving your network the power to **bend**, **curve**, and **transform** the space it operates in.

## Why Use Learning Rates?

Gradient Descent gives you the **direction** to adjust weights and biases, but not how far to go in that direction.

That’s where learning rates come in.

- If the learning rate is too **high**, the model might **overshoot** and bounce around the minimum.
- If it’s too **low**, training will be **painfully slow**, and might even get stuck before reaching the optimal point.

So learning rate is like the **step size**. A good step size balances **speed and stability**.

## How Gradient Descent Works

Every parameter in your neural network (weights and biases) contributes to the loss.

To reduce the loss, we calculate the **partial derivatives** of the loss with respect to each weight or bias.

This tells us how **sensitive** the loss is to each parameter:
- If the derivative is **positive**, then **increasing** that parameter will **increase** the loss → so we want to **decrease** it.
- If the derivative is **negative**, then **increasing** that parameter will **decrease** the loss → so we still subtract the derivative (hence the negative sign in the update rule).

That's why the update rule is:
parameter = parameter - learning_rate * derivative

This is what helps the model **converge** to the lowest loss.

## Manual Overclocking Mechanism

This architecture supports an experimental manual tuning strategy called **Overclocking**.

> ⚠️ This is not part of standard Gradient Descent or Optimizers like Adam. It's a manual intervention technique.

### How Overclocking Works:

1. **Train your model normally** and **log the losses** per epoch.
2. **Copy the model's initial weights** to a second model.
3. On this second model:
   - Use the **same learning rate** and **neurons** and epochs(important! or else the loss logs won't match).
   - Observe **where the loss starts dancing** around a low value (e.g., 0.15) — that’s likely a **local minimum**.
   - At that point (the "overclock moment"), reduce the learning rate by a small amount (like -0.1 or -0.05).
     - This makes the model take **more stable, smaller steps** and helps it settle into a minimum.
   - If instead, you see the model **hovering or plateauing** at a suspicious flat slope, where gradients are very low (saddle point), then **increase** the learning rate slightly.

Overclocking is about **timing** and **loss inspection**, not just tuning a hyperparameter.

---

## Summary

This neural network is not built with high-level abstractions. Every component is included only when it makes mathematical or training sense:

- **Activations** for non-linearity.
- **Learning rate** for stable updates.
- **Gradient descent** for minimizing the loss.
- **Overclocking** for manually adjusting model behavior based on real-time loss patterns.

This README is intentionally written with explanations that *make sense from a builder's point of view*, not just copied from a textbook.

