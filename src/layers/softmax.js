import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

export class SoftmaxLayer {
  type = 'softmax';
  #cache = null;

  forward(X) {
    // y_i = exp(x_i) / sum_j exp(x_j)
    const Y = new Matrix(
      X.data.map((row) => {
        let m = -Infinity;
        for (let j = 0; j < row.length; j++) {
          m = Math.max(m, row[j]);
        }

        const exps = new Array(row.length);
        let s = 0;
        for (let j = 0; j < row.length; j++) {
          const e = Math.exp(row[j] - m);
          exps[j] = e;
          s += e;
        }

        const out = new Array(row.length);
        for (let j = 0; j < row.length; j++) {
          out[j] = exps[j] / s;
        }
        return out;
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

    // For each row:
    // dX = J^T dY, where J = diag(y) - y y^T
    // Efficient form:
    // dX = y * (dY - sum(dY * y))
    const dX = new Matrix(
      Y.data.map((row, i) => {
        const dYRow = dY.data[i];

        let dot = 0;
        for (let j = 0; j < row.length; j++) {
          dot += dYRow[j] * row[j];
        }

        const out = new Array(row.length);
        for (let j = 0; j < row.length; j++) {
          out[j] = row[j] * (dYRow[j] - dot);
        }
        return out;
      }),
    );

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
