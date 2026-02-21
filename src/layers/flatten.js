import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

export class FlattenLayer {
  type = 'flatten';
  #cache = null;

  forward(X) {
    const N = X.length;
    assert(N > 0, 'FlattenLayer.forward: empty batch');

    const C = X[0].length;
    const H = X[0][0].rowDim;
    const W = X[0][0].colDim;
    const D = C * H * W;

    // verify consistent shapes
    for (let n = 0; n < N; n++) {
      assert(
        X[n].length === C,
        `FlattenLayer.forward: expected ${C} channels, got ${X[n].length} at sample ${n}`,
      );
      for (let c = 0; c < C; c++) {
        const M = X[n][c];
        assert(
          M.rowDim === H && M.colDim === W,
          `FlattenLayer.forward: expected ${H}x${W}, got ${M.rowDim}x${M.colDim} at sample ${n} channel ${c}`,
        );
      }
    }

    const Y = Matrix.zeros(N, D);
    for (let n = 0; n < N; n++) {
      const row = Y.data[n];
      let t = 0;

      for (let c = 0; c < C; c++) {
        const M = X[n][c];
        for (let i = 0; i < H; i++) {
          const srcRow = M.data[i];
          for (let j = 0; j < W; j++) {
            row[t++] = srcRow[j];
          }
        }
      }
    }

    this.#cache = { N, C, H, W, D };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'FlattenLayer.backward: cache is null');
    const { N, C, H, W, D } = this.#cache;
    assert(
      dY.rowDim === N && dY.colDim === D,
      `FlattenLayer.backward: expected ${N}x${D}, got ${dY.rowDim}x${dY.colDim}`,
    );

    const dX = new Array(N);
    for (let n = 0; n < N; n++) {
      const sample = new Array(C);
      const row = dY.data[n];
      let t = 0;

      for (let c = 0; c < C; c++) {
        const M = Matrix.zeros(H, W);
        for (let i = 0; i < H; i++) {
          const dstRow = M.data[i];
          for (let j = 0; j < W; j++) {
            dstRow[j] = row[t++];
          }
        }
        sample[c] = M;
      }
      dX[n] = sample;
    }

    this.#cache = null;
    return dX;
  }
}
