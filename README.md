# AI

A from-scratch neural network library in JavaScript — built for learning.

This is an experimental set of ML primitives designed to help understand how
modern AI works. The code is not optimized for performance or memory usage. It
is optimized for readability and understanding. The goal was to map the math to
the code as directly as possible.

The only abstractions are a Vector and 2D Matrix class. No tensor abstractions
were used so that batch logic is clear and explicit. No logits optimizations were
made so that the pipeline stays straightforward.

All math is implemented to be numerically stable. Softmax uses the max-shift
trick to prevent overflow. Sigmoid and tanh are built on a stable softplus
formulation. Loss functions clamp predictions to avoid log(0) and division by
zero.

## Layers

- **Dense** — fully connected layer with weights and biases
- **Conv** — 2D convolution with configurable kernel size and multi-channel support
- **Pool** — max pooling with configurable window size
- **Padding** — zero-padding for spatial layers
- **Flatten** — reshapes multi-channel 2D feature maps into flat vectors
- **Reshape** — converts flat vectors into single-channel 2D feature maps
- **ReLU** — rectified linear activation with optional leaky slope
- **Sigmoid** — logistic activation, maps outputs to (0, 1)
- **Tanh** — hyperbolic tangent activation, maps outputs to (-1, 1)
- **Softmax** — normalizes outputs into a probability distribution

## Loss Functions

- **MSE** — mean squared error for regression tasks
- **BCE** — binary cross-entropy for binary classification
- **SCE** — softmax cross-entropy for multi-class classification

## Training

- **SGD** optimizer with configurable learning rate
- **Xavier** and **He** weight initialization
- **Early stopping** with configurable patience and minimum improvement
- **Batch processing** with configurable batch size and optional shuffling

## Model

Highly configurable base `Model` class with save/load serialization. Factory
methods `mlp()` and `cnn()` handle layer construction, weight initialization,
and activation wiring automatically.

## Example

```js
const model = cnn({
  input: { rows: 28, cols: 28 },
  conv: [
    { out: 16, kernel: 3, pad: 1, pool: 2 },
    { out: 32, kernel: 3, pad: 1, pool: 2 },
  ],
  dense: [32 * 7 * 7, 256, 10],
  activation: 'relu',
  leaky: 0.01,
  final: 'softmax',
  loss: 'sce',
  batchSize: 64,
  learningRate: 0.01,
  epochs: 60,
  patience: 5,
});

model.train(Xtrain, Ttrain);
```

Both the MLP and CNN tests achieve a consistent 98%+ accuracy on MNIST — not
bad for a library written entirely from scratch with no dependencies.

## Tests

Run tests with `just test <name>`:

- `xor` — MLP solves the XOR problem
- `mnist` — MLP trained on MNIST digit classification
- `mnist_cnn` — CNN trained on MNIST digit classification

## Feedback

If you find this useful or learned something from it, a ⭐ would be appreciated!
Feedback, questions, and issues are always welcome.
