"""A small supervised MLP; no dataset download or external training framework."""
import jax
import jax.numpy as jnp
import numpy as np

from dropout import make_dropout, layer_parameters

INPUTS = 32
CLASSES = 4
LEARNING_RATE = 0.1


def dataset(seed=20261021, train_size=2048, validation_size=512):
    rng = np.random.Generator(np.random.PCG64(seed))
    teacher = rng.standard_normal((INPUTS, CLASSES)).astype(np.float32)
    x = rng.standard_normal((train_size + validation_size, INPUTS)).astype(np.float32)
    y = (x @ teacher).argmax(axis=1).astype(np.int32)
    return (x[:train_size], y[:train_size]), (x[train_size:], y[train_size:])


def initialize(width, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for left, right in ((INPUTS, width), (width, width), (width, CLASSES)):
        weights = (rng.standard_normal((left, right)) * np.sqrt(2.0 / left)).astype(np.float32)
        result.append((jnp.asarray(weights), jnp.zeros(right, dtype=jnp.float32)))
    return tuple(result)


def logits(weights, x, *, drop=None, engine=None, root=None, step=None):
    stride = (x.shape[0] * weights[0][1].size + 1) // 2
    for layer, (w, b) in enumerate(weights):
        x = x @ w + b
        if layer < 2:
            x = jax.nn.relu(x)
            if drop is not None:
                p = layer_parameters(engine, root, step, layer, stride)
                x = drop(x, p)
    return x


def cross_entropy(scores, y):
    logp = jax.nn.log_softmax(scores)
    return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=1))


@jax.jit
def evaluate(weights, x, y):
    scores = logits(weights, x)
    return cross_entropy(scores, y), jnp.mean(scores.argmax(axis=1) == y)


def training_step(engine, implementation, rows=32, interpret=False):
    drop = make_dropout(engine, implementation, rows, interpret)

    def loss(weights, x, y, root, step):
        return cross_entropy(logits(weights, x, drop=drop, engine=engine, root=root, step=step), y)

    @jax.jit
    def step(weights, x, y, root, number):
        value, gradients = jax.value_and_grad(loss)(weights, x, y, root, number)
        weights = jax.tree.map(lambda w, g: w - LEARNING_RATE * g, weights, gradients)
        return weights, value
    return step
