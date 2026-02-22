import { assert } from '../lib/assert.js';

export class ReluConvLayer {
  type = 'relu_conv';
  #cache = null;

  constructor({ leaky = 0 } = {}) {
    this.leaky = leaky;
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    return new ReluConvLayer({ leaky: source.leaky });
  }

  forward(X) {
    const N = X.length;
    assert(N > 0, 'ReluConvLayer.forward: empty batch');

    const C = X[0].length;
    const Y = new Array(N);

    for (let n = 0; n < N; n++) {
      const sample = X[n];
      assert(
        sample.length === C,
        `ReluConvLayer.forward: expected ${C} channels, got ${sample.length} at sample ${n}`,
      );

      const outSample = new Array(C);
      for (let c = 0; c < C; c++) {
        const M = sample[c];
        const R = M.map((x) => (x > 0 ? x : x * this.leaky));
        outSample[c] = R;
      }
      Y[n] = outSample;
    }

    this.#cache = { Y };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'ReluConvLayer.backward: cache is null');
    const { Y } = this.#cache;
    const N = dY.length;
    const C = Y[0].length;

    const dX = new Array(N);
    for (let n = 0; n < N; n++) {
      const dYsample = dY[n];
      const Ysample = Y[n];

      assert(
        dYsample.length === C,
        `ReluConvLayer.backward: expected ${C} channels, got ${dYsample.length} at sample ${n}`,
      );

      const outSample = new Array(C);
      for (let c = 0; c < C; c++) {
        const G = dYsample[c];
        const A = Ysample[c];
        const dM = G.mul(A.map((y) => (y > 0 ? 1 : this.leaky)));
        outSample[c] = dM;
      }
      dX[n] = outSample;
    }

    this.#cache = null;
    return dX;
  }
}
