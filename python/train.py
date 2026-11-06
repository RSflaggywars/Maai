#!/usr/bin/env python3
"""AI-Lab trainer - teaches a tiny neural net with numpy, exports weights.

The network: 2 inputs -> H hidden (tanh) -> 1 output (sigmoid).
Trained on XOR (the classic "can a net learn anything?" test).

Run:  python train.py
Output: ../weights.txt  (plain text - the C side reads this)
"""

import os
import numpy as np

HIDDEN = 8
EPOCHS = 20000
LR = 1.0
SEED = 42

# XOR dataset: inputs -> target
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
Y = np.array([[0], [1], [1], [0]], dtype=np.float64)


def init():
    rng = np.random.default_rng(SEED)
    # Xavier-ish init
    W1 = rng.uniform(-1, 1, (2, HIDDEN)) * np.sqrt(2 / 2)
    b1 = np.zeros((1, HIDDEN))
    W2 = rng.uniform(-1, 1, (HIDDEN, 1)) * np.sqrt(2 / HIDDEN)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


def forward(W1, b1, W2, b2, X):
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = 1 / (1 + np.exp(-Z2))
    return Z1, A1, Z2, A2


def train():
    W1, b1, W2, b2 = init()
    for epoch in range(EPOCHS):
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        # backprop
        dZ2 = (A2 - Y) / len(X)
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (1 - A1 ** 2)          # tanh'
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)
        # update
        W2 -= LR * dW2
        b2 -= LR * db2
        W1 -= LR * dW1
        b1 -= LR * db1
        if epoch % 5000 == 0 or epoch == EPOCHS - 1:
            loss = float(np.mean((A2 - Y) ** 2))
            print(f"epoch {epoch:6d}  loss {loss:.6f}")
    return W1, b1, W2, b2


def save(W1, b1, W2, b2, path):
    """Plain-text format the C program parses:
        2 8          <- W1 dims (in out)
        <16 floats>  <- W1 row-major
        1 8          <- b1 dims (1 x hidden)
        <8 floats>
        8 1          <- W2 dims
        <8 floats>
        1 1          <- b2 dims
        <1 float>
    """
    with open(path, "w") as f:
        f.write(f"{W1.shape[0]} {W1.shape[1]}\n")
        f.write(" ".join(f"{v:.8f}" for v in W1.flatten()) + "\n")
        f.write(f"{b1.shape[0]} {b1.shape[1]}\n")
        f.write(" ".join(f"{v:.8f}" for v in b1.flatten()) + "\n")
        f.write(f"{W2.shape[0]} {W2.shape[1]}\n")
        f.write(" ".join(f"{v:.8f}" for v in W2.flatten()) + "\n")
        f.write(f"{b2.shape[0]} {b2.shape[1]}\n")
        f.write(" ".join(f"{v:.8f}" for v in b2.flatten()) + "\n")


def main():
    W1, b1, W2, b2 = train()
    out = os.path.join(os.path.dirname(__file__), "..", "weights.txt")
    save(W1, b1, W2, b2, out)

    # show what the net learned
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    print("\nXOR results:")
    for x, y, p in zip(X, Y.flatten(), A2.flatten()):
        print(f"  {int(x[0])} XOR {int(x[1])} = {int(y)}  ->  {p:.3f}")
    print(f"\nWeights saved to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
