import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

export class SoftmaxLayer {
  type = 'softmax';
  #cache = null;

  forward(X) {
    // y_i = exp(x_i - max) / sum(exp(x_j - max))
    const Y = new Matrix(
      X.data.map((row) => {
        const m = Math.max(...row);
        const exps = row.map((x) => Math.exp(x - m));
        const s = exps.reduce((a, b) => a + b, 0);
        return exps.map((e) => e / s);
      }),
    );

    this.#cache = { Y };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'SoftmaxLayer.backward: cache is null');
    const { Y } = this.#cache;
    assert(
      dY.rowDim === Y.rowDim && dY.colDim === Y.colDim,
      `SoftmaxLayer.backward: expected dY shape ${Y.rowDim}x${Y.colDim}, got ${dY.rowDim}x${dY.colDim}`,
    );

    // dX = Y * (dY - sum(dY * Y))
    const dots = Y.mul(dY).sumCols();
    const dX = Y.mul(dY.map((x, i) => x - dots.data[i]));

    this.#cache = null;
    return dX;
  }

  finalize(row) {
    let bestJ = 0;
    let bestV = row[0];
    for (let j = 1; j < row.length; j++) {
      const v = row[j];
      if (v > bestV) {
        bestV = v;
        bestJ = j;
      }
    }
    return bestJ;
  }
}
