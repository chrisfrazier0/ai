import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

// batch   = Matrix[][]   array of samples
// sample  = Matrix[]     array of channels
// channel = Matrix       height x width

export class ReshapeLayer {
  type = 'reshape';
  #cache = null;

  constructor({ rows, cols }) {
    this.rows = rows;
    this.cols = cols;
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = new ReshapeLayer({});
    Object.assign(result, source);
    return result;
  }

  forward(X) {
    const N = X.rowDim;
    const D = X.colDim;
    const H = this.rows;
    const W = this.cols;
    const HW = H * W;

    assert(D === HW, `ReshapeLayer.forward: expected ${HW} columns, got ${D}`);

    const out = new Array(N);
    for (let n = 0; n < N; n++) {
      const flat = X.data[n];

      const M = Matrix.zeros(H, W);
      for (let i = 0; i < H; i++) {
        const row = M.data[i];
        const off = i * W;
        for (let j = 0; j < W; j++) {
          row[j] = flat[off + j];
        }
      }

      out[n] = [M];
    }

    this.#cache = { N, H, W };
    return out;
  }

  backward(dY) {
    assert(this.#cache, 'ReshapeLayer.backward: cache is null');
    const { N, H, W } = this.#cache;
    const HW = H * W;
    assert(
      dY.length === N,
      `ReshapeLayer.backward: expected batch length ${N}, got ${dY.length}`,
    );

    const dX = Matrix.zeros(N, HW);
    for (let n = 0; n < N; n++) {
      const sample = dY[n];
      assert(
        sample.length === 1,
        `ReshapeLayer.backward: expected 1 channel, got ${sample.length}`,
      );

      const dM = sample[0];
      assert(
        dM.rowDim === H && dM.colDim === W,
        `ReshapeLayer.backward: expected channel matrix ${H}x${W}, got ${dM.rowDim}x${dM.colDim}`,
      );

      const row = dX.data[n];
      for (let i = 0; i < H; i++) {
        const off = i * W;
        for (let j = 0; j < W; j++) {
          row[off + j] = dM.data[i][j];
        }
      }
    }

    this.#cache = null;
    return dX;
  }
}
