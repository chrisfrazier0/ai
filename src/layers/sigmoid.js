import { assert } from '../lib/assert.js';
import { sigmoid } from '../lib/math.js';

export class SigmoidLayer {
  type = 'sigmoid';
  #cache = null;

  forward(X) {
    // y = 1 / (1 + e^-x)
    const Y = X.map((x) => sigmoid(x));
    this.#cache = { Y };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'SigmoidLayer.backward: cache is null');
    const { Y } = this.#cache;
    assert(
      dY.rowDim === Y.rowDim && dY.colDim === Y.colDim,
      `SigmoidLayer.backward: expected dY shape ${Y.rowDim}x${Y.colDim}, got ${dY.rowDim}x${dY.colDim}`,
    );

    // dY/dX = Y*(1 - Y)
    const dSig = Y.mul(Y.oneMinus());
    const dX = dY.mul(dSig);

    this.#cache = null;
    return dX;
  }

  finalize(Z) {
    return Z.map((z) => (z >= 0.5 ? 1 : 0));
  }
}
