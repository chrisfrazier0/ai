import { ConvLayer } from './layers/conv.js';
import { DenseLayer } from './layers/dense.js';
import { FlattenLayer } from './layers/flatten.js';
import { PaddingLayer } from './layers/padding.js';
import { PoolLayer } from './layers/pool.js';
import { ReluLayer } from './layers/relu.js';
import { ReluConvLayer } from './layers/relu_conv.js';
import { ReshapeLayer } from './layers/reshape.js';
import { SigmoidLayer } from './layers/sigmoid.js';
import { SoftmaxLayer } from './layers/softmax.js';
import { TanhLayer } from './layers/tanh.js';
import { assert } from './lib/assert.js';
import { Matrix } from './lib/matrix.js';
import { bceLoss } from './loss/bce.js';
import { mseLoss } from './loss/mse.js';
import { sceLoss } from './loss/sce.js';
import { Model } from './model.js';

const activationLookup = {
  sigmoid: SigmoidLayer,
  tanh: TanhLayer,
  relu: ReluLayer,
};

const finalLookup = {
  sigmoid: SigmoidLayer,
  softmax: SoftmaxLayer,
};

const lossLookup = {
  mse: mseLoss,
  bce: bceLoss,
  sce: sceLoss,
};

export function mlp({
  activation,
  leaky = 0,
  final = null,
  loss,
  layers,
  ...args
}) {
  assert(
    activationLookup[activation],
    'mlp: unknown activation type ' + activation,
  );
  assert(
    !final || finalLookup[final],
    'mlp: unknown final activation type' + final,
  );
  assert(lossLookup[loss], 'mlp: unknown loss type ' + loss);
  assert(layers.length > 1, 'mlp: invalid layer definition');

  const init =
    activation === 'relu'
      ? (i, _) => Math.sqrt(2 / i) // He
      : (i, o) => Math.sqrt(2 / (i + o)); // Xavier

  const l = [];
  for (let i = 0; i < layers.length - 1; i++) {
    const inDim = layers[i];
    const outDim = layers[i + 1];
    const std = init(inDim, outDim);
    const bias = activation === 'relu' && i === 0 ? 0.01 : 0;

    l.push(new DenseLayer({ weights: Matrix.randn(inDim, outDim, std), bias }));
    if (i !== layers.length - 2)
      l.push(new activationLookup[activation]({ leaky }));
  }
  if (final) l.push(new finalLookup[final]({ leaky }));

  return new Model({
    ...args,
    loss: lossLookup[loss],
    layers: l,
  });
}

export function cnn({
  input, // { rows, cols }
  conv, // [{ out, kernel=3, pad=0, pool=0, std?, bias? }, ...]
  dense, // dense layer spec
  activation = 'relu',
  leaky = 0,
  final = null,
  loss,
  ...args
}) {
  assert(input.rows && input.cols, 'cnn: invalid input definition');
  assert(conv.length > 0, 'cnn: invalid conv definition');
  assert(dense.length > 0, 'cnn: invalid dense definition');
  assert(
    activationLookup[activation],
    'cnn: unknown activation type ' + activation,
  );
  assert(
    !final || finalLookup[final],
    'cnn: unknown final activation type ' + final,
  );
  assert(lossLookup[loss], 'cnn: unknown loss type ' + loss);

  const layers = [];
  layers.push(new ReshapeLayer({ rows: input.rows, cols: input.cols }));

  let inChannels = 1;
  for (let i = 0; i < conv.length; i++) {
    const block = conv[i];
    const out = block.out;
    const kernel = block.kernel ?? 3;
    const pad = block.pad ?? 0;
    const pool = block.pool ?? 0;
    assert(out, `cnn: conv[${i}] missing 'out'`);

    if (pad) layers.push(new PaddingLayer({ pad }));

    // He-ish init for conv (only using relu activations)
    const std = block.std ?? Math.sqrt(2 / (kernel * kernel * inChannels));
    const bias = block.bias ?? 0;

    layers.push(
      new ConvLayer({
        inChannels,
        outChannels: out,
        kernel,
        std,
        bias,
      }),
    );

    layers.push(new ReluConvLayer({ leaky }));
    if (pool) layers.push(new PoolLayer({ size: pool }));

    inChannels = out;
  }

  layers.push(new FlattenLayer());

  const init =
    activation === 'relu'
      ? (i, _) => Math.sqrt(2 / i) // He
      : (i, o) => Math.sqrt(2 / (i + o)); // Xavier

  for (let i = 0; i < dense.length - 1; i++) {
    const inDim = dense[i];
    const outDim = dense[i + 1];
    const std = init(inDim, outDim);
    const bias = activation === 'relu' && i === 0 ? 0.01 : 0;

    layers.push(
      new DenseLayer({ weights: Matrix.randn(inDim, outDim, std), bias }),
    );
    if (i !== dense.length - 2) {
      layers.push(new activationLookup[activation]({ leaky }));
    }
  }
  if (final) layers.push(new finalLookup[final]({ leaky }));

  return new Model({
    ...args,
    loss: lossLookup[loss],
    layers,
  });
}
