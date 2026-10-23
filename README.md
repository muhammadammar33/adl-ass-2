# Assignment 2 — Neural Networks from Scratch

Everything is implemented in plain NumPy. No deep-learning framework is used.

| File | Contents |
|---|---|
| `nnlib.py` | Library: Dense layer, Sigmoid / ReLU / Linear / Softmax activations, `MLP` with backpropagation, MSE / BCE / Softmax-CE losses, GD / SGD / Momentum optimisers, training loop, numerical gradient check, toy datasets |
| `Assignment2.ipynb` | Derivations, hand calculations and experiments (already run, outputs included) |
| `Assignment2.html` | Static export of the executed notebook (open it in any browser) |

## What is covered

1. **Models**
   - Fully-connected NN with sigmoid activation (two-moons) — §5.1
   - 2-layer ReLU network for XOR: exact analytical solution plus a trained version — §5.2
   - Gradient-based learning for XOR with sigmoids, with the training loop written out explicitly — §5.3
   - MLP for multi-class classification with a softmax output (3-class spiral) — §5.4
   - General deep MLP (regression with MSE) — §1, §5.5
2. **Losses:** MSE, binary cross-entropy, softmax cross-entropy, each worked out by hand — §2
3. **Gradient derivations:** dL/dŷ and dL/dz for each loss, including the simplifications σ+BCE → ŷ−y and softmax+CE → p−y — §3
4. **Backpropagation:** matrix-form derivation, a full forward/backward/update pass computed by hand for a 2-2-1 network, and finite-difference gradient checks — §4.1–4.3
   **Optimisers:** GD, SGD and SGD with Momentum, with their update rules, hand-computed iterations and empirical comparisons — §4.4–4.7

## Running

```bash
pip install -r requirements.txt
jupyter notebook Assignment2.ipynb
```
