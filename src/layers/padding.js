import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

export class PaddingLayer {
  type = 'padding';
  #cache = null;

  constructor({ pad = 1 } = {}) {
    this.pad = pad;
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = new PaddingLayer({ pad: source.pad });
    return result;
  }

  forward(X) {
    const p = this.pad;
    const N = X.length;
    const C = N > 0 ? X[0].length : 0;

    // cache original spatial dims per channel (assumes consistent across batch)
    const H = C > 0 ? X[0][0].rowDim : 0;
    const W = C > 0 ? X[0][0].colDim : 0;

    // verify consistent shapes
    for (let n = 0; n < N; n++) {
      assert(
        X[n].length === C,
        `PaddingLayer.forward: expected ${C} channels, got ${X[n].length} at sample ${n}`,
      );
      for (let c = 0; c < C; c++) {
        const M = X[n][c];
        assert(
          M.rowDim === H && M.colDim === W,
          `PaddingLayer.forward: expected ${H}x${W}, got ${M.rowDim}x${M.colDim} at sample ${n} channel ${c}`,
        );
      }
    }

    // no-op padding
    if (p === 0) {
      this.#cache = { N, C, H, W, p };
      return X;
    }

    // standard padding
    const Y = new Array(N);
    for (let n = 0; n < N; n++) {
      const sample = X[n];
      const outSample = new Array(C);

      for (let c = 0; c < C; c++) {
        const M = sample[c];
        const P = Matrix.zeros(H + 2 * p, W + 2 * p);

        for (let i = 0; i < H; i++) {
          const srcRow = M.data[i];
          const dstRow = P.data[i + p];
          for (let j = 0; j < W; j++) {
            dstRow[j + p] = srcRow[j];
          }
        }

        outSample[c] = P;
      }

      Y[n] = outSample;
    }

    this.#cache = { N, C, H, W, p };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'PaddingLayer.backward: cache is null');
    const { N, C, H, W, p } = this.#cache;
    assert(
      dY.length === N,
      `PaddingLayer.backward: expected batch length ${N}, got ${dY.length}`,
    );

    // no-op padding
    if (p === 0) {
      this.#cache = null;
      return dY;
    }

    const dX = new Array(N);
    for (let n = 0; n < N; n++) {
      const sample = dY[n];
      assert(
        sample.length === C,
        `PaddingLayer.backward: expected ${C} channels, got ${sample.length} at sample ${n}`,
      );

      const outSample = new Array(C);
      for (let c = 0; c < C; c++) {
        const G = sample[c];
        assert(
          G.rowDim === H + 2 * p && G.colDim === W + 2 * p,
          `PaddingLayer.backward: expected ${H + 2 * p}x${W + 2 * p}, got ${G.rowDim}x${G.colDim} at sample ${n} channel ${c}`,
        );

        const M = Matrix.zeros(H, W);
        for (let i = 0; i < H; i++) {
          const dstRow = M.data[i];
          const srcRow = G.data[i + p];
          for (let j = 0; j < W; j++) {
            dstRow[j] = srcRow[j + p];
          }
        }

        outSample[c] = M;
      }

      dX[n] = outSample;
    }

    this.#cache = null;
    return dX;
  }
}
