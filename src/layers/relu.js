import { assert } from '../lib/assert.js';

export class ReluLayer {
  type = 'relu';
  #cache = null;

  constructor({ leaky = 0 } = {}) {
    this.leaky = leaky;
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = new ReluLayer();
    Object.assign(result, source);
    return result;
  }

  forward(X) {
    // Y = | x > 0: x
    //     | x <= 0: x * λ
    const Y = X.map((x) => (x > 0 ? x : x * this.leaky));
    this.#cache = { X };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'ReluLayer.backward: cache is null');
    const { X } = this.#cache;
    assert(
      dY.rowDim === X.rowDim && dY.colDim === X.colDim,
      `ReluLayer.backward: expected dY shape ${X.rowDim}x${X.colDim}, got ${dY.rowDim}x${dY.colDim}`,
    );

    // dY/dX = | x > 0: 1
    //         | x <= 0: λ
    const dRel = X.map((x) => (x > 0 ? 1 : this.leaky));
    const dX = dY.mul(dRel);

    this.#cache = null;
    return dX;
  }
}
