"""
nnlib.py - a small neural-network library written from scratch with NumPy.

Everything needed for Assignment 2 lives here:
  * activations  : Sigmoid, ReLU, Identity, Softmax          (forward + backward)
  * layers       : Dense (fully-connected)                     (forward + backward)
  * model        : MLP  (stack of Dense + activation pairs)    (forward + backprop)
  * losses       : MSE, BinaryCrossEntropy, SoftmaxCrossEntropy (value + gradient)
  * optimizers   : GD, SGD, Momentum
  * utilities    : training loop, numerical gradient check, toy datasets

Conventions
-----------
Samples are stored as rows: X has shape (N, d_in), the output of a layer has
shape (N, d_out). A Dense layer computes Z = X W + b with W of shape
(d_in, d_out) and b of shape (d_out,). Every loss is averaged over the N
samples of the batch, so all gradients carry a 1/N factor.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
class Sigmoid:
    """a = 1 / (1 + e^{-z}),  da/dz = a (1 - a)."""

    def forward(self, z):
        # numerically stable form: never evaluates exp of a large positive number
        a = np.empty_like(z, dtype=float)
        pos = z >= 0
        a[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        a[~pos] = ez / (1.0 + ez)
        self.a = a
        return a

    def backward(self, grad_a):
        return grad_a * self.a * (1.0 - self.a)


class ReLU:
    """a = max(0, z),  da/dz = 1[z > 0]."""

    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, grad_a):
        return grad_a * self.mask


class Identity:
    """a = z (used for the linear output of a regression network)."""

    def forward(self, z):
        return z

    def backward(self, grad_a):
        return grad_a


class Softmax:
    """a_k = e^{z_k} / sum_j e^{z_j}  (applied row-wise).

    Jacobian: da_i/dz_j = a_i (delta_ij - a_j), hence the vector-Jacobian product
    dL/dz = a * (dL/da - sum_k dL/da_k a_k).
    """

    def forward(self, z):
        z = z - z.max(axis=1, keepdims=True)  # shift for numerical stability
        e = np.exp(z)
        self.a = e / e.sum(axis=1, keepdims=True)
        return self.a

    def backward(self, grad_a):
        s = np.sum(grad_a * self.a, axis=1, keepdims=True)
        return self.a * (grad_a - s)


ACTIVATIONS = {"sigmoid": Sigmoid, "relu": ReLU, "linear": Identity, "softmax": Softmax}


# ---------------------------------------------------------------------------
# Fully-connected layer
# ---------------------------------------------------------------------------
class Dense:
    """Z = X W + b.

    Backward pass, given G = dL/dZ (shape N x d_out):
        dL/dW = X^T G,   dL/db = sum_n G_n,   dL/dX = G W^T
    """

    def __init__(self, d_in, d_out, rng, init="xavier"):
        if init == "he":  # good for ReLU
            scale = np.sqrt(2.0 / d_in)
        else:  # Xavier/Glorot, good for sigmoid / tanh
            scale = np.sqrt(1.0 / d_in)
        self.W = rng.normal(0.0, scale, size=(d_in, d_out))
        self.b = np.zeros(d_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_z):
        self.dW = self.x.T @ grad_z
        self.db = grad_z.sum(axis=0)
        return grad_z @ self.W.T


# ---------------------------------------------------------------------------
# Multi-layer perceptron
# ---------------------------------------------------------------------------
class MLP:
    """A fully-connected feed-forward network.

    Parameters
    ----------
    sizes       : list of layer widths, e.g. [2, 4, 1]
    activations : one activation name per Dense layer, e.g. ["relu", "sigmoid"]
    """

    def __init__(self, sizes, activations, seed=0, init=None):
        assert len(activations) == len(sizes) - 1
        rng = np.random.default_rng(seed)
        self.layers, self.acts = [], []
        for d_in, d_out, act in zip(sizes[:-1], sizes[1:], activations):
            layer_init = init or ("he" if act == "relu" else "xavier")
            self.layers.append(Dense(d_in, d_out, rng, layer_init))
            self.acts.append(ACTIVATIONS[act]())

    def forward(self, x):
        a = x
        for layer, act in zip(self.layers, self.acts):
            a = act.forward(layer.forward(a))
        return a

    __call__ = forward

    def backward(self, grad_out):
        """Back-propagate dL/d(output) through every layer (stores dW, db)."""
        g = grad_out
        for layer, act in zip(reversed(self.layers), reversed(self.acts)):
            g = layer.backward(act.backward(g))
        return g

    def params(self):
        """Parameters and their gradients, as matching lists of arrays."""
        ps, gs = [], []
        for layer in self.layers:
            ps += [layer.W, layer.b]
            gs += [layer.dW, layer.db]
        return ps, gs

    def predict(self, x):
        out = self.forward(x)
        if out.shape[1] == 1:
            return (out[:, 0] >= 0.5).astype(int)
        return out.argmax(axis=1)


# ---------------------------------------------------------------------------
# Loss functions (all averaged over the batch)
# ---------------------------------------------------------------------------
EPS = 1e-12


class MSE:
    """L = 1/N sum_n sum_k (yhat_nk - y_nk)^2,   dL/dyhat = 2/N (yhat - y)."""

    def __call__(self, yhat, y):
        return np.sum((yhat - y) ** 2) / yhat.shape[0]

    def grad(self, yhat, y):
        return 2.0 * (yhat - y) / yhat.shape[0]


class BinaryCrossEntropy:
    """L = -1/N sum_n [y log yhat + (1-y) log(1-yhat)],
    dL/dyhat = 1/N (yhat - y) / (yhat (1 - yhat)).
    """

    def __call__(self, yhat, y):
        p = np.clip(yhat, EPS, 1 - EPS)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / yhat.shape[0]

    def grad(self, yhat, y):
        p = np.clip(yhat, EPS, 1 - EPS)
        return (p - y) / (p * (1 - p)) / yhat.shape[0]


class SoftmaxCrossEntropy:
    """L = -1/N sum_n sum_k y_nk log p_nk  (y one-hot, p = softmax output),
    dL/dp = -1/N y / p.  Combined with the softmax Jacobian: dL/dz = 1/N (p - y).
    """

    def __call__(self, p, y):
        return -np.sum(y * np.log(np.clip(p, EPS, 1.0))) / p.shape[0]

    def grad(self, p, y):
        return -y / np.clip(p, EPS, 1.0) / p.shape[0]


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
class GD:
    """theta <- theta - lr * grad   (full batch: the trainer passes all data)."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class SGD(GD):
    """Same update rule as GD but applied to the gradient of a mini-batch."""


class Momentum:
    """v <- beta v + grad,   theta <- theta - lr v   (heavy-ball momentum)."""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.beta
            v += g
            p -= self.lr * v


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train(model, loss_fn, optimizer, X, Y, epochs=1000, batch_size=None, seed=0,
          verbose=0):
    """Train `model` in place and return the per-epoch loss on the full data.

    batch_size=None  -> full-batch Gradient Descent (one update per epoch)
    batch_size=1     -> pure Stochastic Gradient Descent
    batch_size=B     -> mini-batch SGD
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    B = N if batch_size is None else batch_size
    history = []
    for epoch in range(epochs):
        idx = rng.permutation(N) if B < N else np.arange(N)
        for start in range(0, N, B):
            batch = idx[start:start + B]
            out = model.forward(X[batch])
            model.backward(loss_fn.grad(out, Y[batch]))
            optimizer.step(*model.params())
        history.append(loss_fn(model.forward(X), Y))
        if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:5d}  loss {history[-1]:.6f}")
    return np.array(history)


def numerical_gradient(f, x, h=1e-6):
    """Central-difference estimate of df/dx for every entry of array x."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        grad[i] = (fp - fm) / (2 * h)
    return grad


def gradient_check(model, loss_fn, X, Y):
    """Compare backprop gradients with numerical ones; return max relative error."""
    model.backward(loss_fn.grad(model.forward(X), Y))
    params, grads = model.params()
    errors = []
    for p, g in zip(params, grads):
        num = numerical_gradient(lambda: loss_fn(model.forward(X), Y), p)
        errors.append(np.linalg.norm(num - g) / max(np.linalg.norm(num) + np.linalg.norm(g), 1e-12))
    return max(errors)


def one_hot(labels, n_classes):
    out = np.zeros((labels.size, n_classes))
    out[np.arange(labels.size), labels] = 1.0
    return out


def accuracy(model, X, labels):
    return np.mean(model.predict(X) == labels)


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------
def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


def make_moons(n=400, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    upper = np.c_[np.cos(t1), np.sin(t1)]
    lower = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n1), np.ones(n - n1)].astype(int)
    return X, y


def make_spiral(n_per_class=150, n_classes=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(n_classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, k))
    return np.vstack(X), np.concatenate(y)
