import { assert } from '../lib/assert.js';
import { tanh } from '../lib/math.js';

export class TanhLayer {
  type = 'tanh';
  #cache = null;

  forward(X) {
    // y = (e^x - e^-x) / (e^x + e^-x)
    const Y = X.map((x) => tanh(x));
    this.#cache = { Y };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'TanhLayer.backward: cache is null');
    const { Y } = this.#cache;
    assert(
      dY.rowDim === Y.rowDim && dY.colDim === Y.colDim,
      `TanhLayer.backward: expected dY shape ${Y.rowDim}x${Y.colDim}, got ${dY.rowDim}x${dY.colDim}`,
    );

    // dY/dX = 1 - Y^2
    const dTanh = Y.mul(Y).oneMinus();
    const dX = dY.mul(dTanh);

    this.#cache = null;
    return dX;
  }
}
