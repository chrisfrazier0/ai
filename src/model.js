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
import { bceLoss } from './loss/bce.js';
import { mseLoss } from './loss/mse.js';
import { sceLoss } from './loss/sce.js';

const lossLookup = {
  mse: mseLoss,
  bce: bceLoss,
  softmax_ce: sceLoss,
};

export class Model {
  constructor({
    loss,
    batchSize = 1,
    learningRate = 1e-3,
    epochs = 1,
    minImprovement = 0,
    patience = Infinity,
    shuffle = true,
    layers = [],
  }) {
    this.lossFn = loss;
    this.loss = loss.typeTag;
    this.batchSize = batchSize;
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.minImprovement = minImprovement;
    this.patience = patience;
    this.shuffle = shuffle;
    this.layers = layers;
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = Object.create(Model.prototype);
    Object.assign(result, source);

    result.lossFn = lossLookup[result.loss];
    result.layers = result.layers.map((layer) => {
      switch (layer.type) {
        case 'conv':
          return ConvLayer.fromJSON(layer);
        case 'dense':
          return DenseLayer.fromJSON(layer);
        case 'flatten':
          return new FlattenLayer();
        case 'padding':
          return PaddingLayer.fromJSON(layer);
        case 'pool':
          return PoolLayer.fromJSON(layer);
        case 'relu_conv':
          return ReluConvLayer.fromJSON(layer);
        case 'relu':
          return ReluLayer.fromJSON(layer);
        case 'reshape':
          return ReshapeLayer.fromJSON(layer);
        case 'sigmoid':
          return new SigmoidLayer();
        case 'softmax':
          return new SoftmaxLayer();
        case 'tanh':
          return new TanhLayer();
        default:
          throw new Error('Model#fromJSON: unknown layer type ' + layer.type);
      }
    });
    return result;
  }

  forward(X) {
    let Y = X;
    for (const layer of this.layers) {
      Y = layer.forward(Y);
    }
    return Y;
  }

  backward(dY) {
    let d = dY;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      d = this.layers[i].backward(d, this.learningRate);
    }
    return d;
  }

  finalize(Z) {
    const last = this.layers.length - 1;
    const layer = this.layers[last];
    return layer.finalize ? layer.finalize(Z) : Z;
  }

  getLoss(X, T) {
    const Y = this.forward(X);
    return this.lossFn(Y, T).loss;
  }

  getState() {
    return this.layers.map((l) =>
      l.getState && l.setState ? l.getState() : null,
    );
  }

  setState(state) {
    assert(
      state.length === this.layers.length,
      `Model.setState: expected ${this.layers.length}, got ${state.length}`,
    );
    this.layers.forEach(
      (l, i) => l.getState && l.setState && l.setState(state[i]),
    );
    return this;
  }

  train(X, T, v = 0, log = false) {
    assert(
      T.rowDim === X.rowDim,
      `Model.train: expected ${X.rowDim} targets, got ${T.rowDim}`,
    );

    const fmt = (x) => (Number.isFinite(x) ? x.toFixed(6) : String(x));
    const N = X.rowDim;
    const B = Math.min(Math.max(this.batchSize, 1), N);

    const idxAll = Array.from({ length: N }, (_, i) => i);
    const shuffle = (arr) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    };

    const useVal = v !== 0;
    const vFrac = useVal ? Math.min(v, 0.9) : 0;
    const vCount = useVal ? Math.floor(N * vFrac) : 0;

    if (this.shuffle) shuffle(idxAll);
    const valIdx = useVal ? idxAll.slice(0, vCount) : null;
    const trainIdx = useVal ? idxAll.slice(vCount) : idxAll;

    assert(
      !useVal || trainIdx.length > 0,
      'Model.train: validation split leaves no training data',
    );

    const Xv = useVal ? X.takeRows(valIdx) : null;
    const Tv = useVal ? T.takeRows(valIdx) : null;

    let bestLoss = Infinity;
    let bestState = null;
    let badEpochs = 0;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let epochLossSum = 0;
      let seen = 0;

      if (this.shuffle) shuffle(trainIdx);

      for (let start = 0; start < trainIdx.length; start += B) {
        const end = Math.min(start + B, trainIdx.length);
        const batchIdx = trainIdx.slice(start, end);

        const Xb = X.takeRows(batchIdx);
        const Tb = T.takeRows(batchIdx);

        const Yb = this.forward(Xb);
        assert(
          Yb.colDim === Tb.colDim,
          `Model.train: expected ${T.colDim} output columns, got ${Yb.colDim}`,
        );

        const { loss, dY } = this.lossFn(Yb, Tb);
        this.backward(dY);

        epochLossSum += loss * (end - start);
        seen += end - start;
      }

      const trainLoss = epochLossSum / Math.max(seen, 1);
      const epochLoss = useVal ? this.getLoss(Xv, Tv) : trainLoss;

      const improved =
        bestLoss === Infinity ||
        (bestLoss - epochLoss) / Math.max(Math.abs(bestLoss), 1e-12) >=
          this.minImprovement;

      if (improved) {
        bestLoss = epochLoss;
        bestState = this.getState();
        badEpochs = 0;
      } else {
        badEpochs++;
        if (badEpochs >= this.patience) {
          if (bestState) this.setState(bestState);
          return {
            earlyStop: true,
            epochs: epoch + 1,
            bestLoss,
            epochLoss,
          };
        }
      }

      if (log)
        console.log(
          `epoch ${epoch + 1}/${this.epochs}` +
            `  train=${fmt(trainLoss)}` +
            (useVal ? `  val=${fmt(epochLoss)}` : '') +
            `  best=${fmt(bestLoss)}` +
            `  bad=${badEpochs}/${this.patience === Infinity ? 'inf' : this.patience}`,
        );
    }

    return {
      earlyStop: false,
      epochs: this.epochs,
      bestLoss,
    };
  }
}
