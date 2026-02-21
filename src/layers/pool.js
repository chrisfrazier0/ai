import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

export class PoolLayer {
  type = 'pool';
  #cache = null;

  constructor({ size = 2 } = {}) {
    this.size = size;
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = new PoolLayer({ size: source.size });
    return result;
  }

  forward(X) {
    const p = this.size;
    const N = X.length;
    assert(N > 0, 'PoolLayer.forward: empty batch');

    const C = X[0].length;
    const H = X[0][0].rowDim;
    const W = X[0][0].colDim;

    // verify consistent shapes
    for (let n = 0; n < N; n++) {
      assert(
        X[n].length === C,
        `PoolLayer.forward: expected ${C} channels, got ${X[n].length} at sample ${n}`,
      );
      for (let c = 0; c < C; c++) {
        const M = X[n][c];
        assert(
          M.rowDim === H && M.colDim === W,
          `PoolLayer.forward: expected ${H}x${W}, got ${M.rowDim}x${M.colDim} at sample ${n} channel ${c}`,
        );
      }
    }

    const outH = Math.floor(H / p);
    const outW = Math.floor(W / p);
    assert(
      outH > 0 && outW > 0,
      `PoolLayer.forward: pool ${p} too large for ${H}x${W}`,
    );

    const Y = new Array(N);

    // cache argmax positions for each output cell
    // stored as two matrices per (n,c)
    // maxI: outH x outW, maxJ: outH x outW
    const maxI = new Array(N);
    const maxJ = new Array(N);

    for (let n = 0; n < N; n++) {
      const sample = X[n];
      const outSample = new Array(C);
      maxI[n] = new Array(C);
      maxJ[n] = new Array(C);

      for (let c = 0; c < C; c++) {
        const M = sample[c];

        const Yc = Matrix.zeros(outH, outW);
        const Ii = Matrix.zeros(outH, outW);
        const Ij = Matrix.zeros(outH, outW);

        for (let i = 0; i < outH; i++) {
          const yRow = Yc.data[i];
          const iiRow = Ii.data[i];
          const ijRow = Ij.data[i];

          const baseI = i * p;

          for (let j = 0; j < outW; j++) {
            const baseJ = j * p;

            let best = -Infinity;
            let bi = baseI;
            let bj = baseJ;

            // search p×p window
            for (let di = 0; di < p; di++) {
              const srcRow = M.data[baseI + di];
              for (let dj = 0; dj < p; dj++) {
                const v = srcRow[baseJ + dj];
                if (v > best) {
                  best = v;
                  bi = baseI + di;
                  bj = baseJ + dj;
                }
              }
            }

            yRow[j] = best;
            iiRow[j] = bi;
            ijRow[j] = bj;
          }
        }

        outSample[c] = Yc;
        maxI[n][c] = Ii;
        maxJ[n][c] = Ij;
      }

      Y[n] = outSample;
    }

    this.#cache = { N, C, H, W, outH, outW, maxI, maxJ };
    return Y;
  }

  backward(dY) {
    assert(this.#cache, 'PoolLayer.backward: cache is null');
    const { N, C, H, W, outH, outW, maxI, maxJ } = this.#cache;

    assert(
      dY.length === N,
      `PoolLayer.backward: expected batch length ${N}, got ${dY.length}`,
    );

    const dX = new Array(N);
    for (let n = 0; n < N; n++) {
      const dYsample = dY[n];
      assert(
        dYsample.length === C,
        `PoolLayer.backward: expected ${C} channels, got ${dYsample.length} at sample ${n}`,
      );

      const outSample = new Array(C);
      for (let c = 0; c < C; c++) {
        const G = dYsample[c];
        assert(
          G.rowDim === outH && G.colDim === outW,
          `PoolLayer.backward: expected ${outH}x${outW}, got ${G.rowDim}x${G.colDim} at sample ${n} channel ${c}`,
        );

        const dM = Matrix.zeros(H, W);

        const Ii = maxI[n][c];
        const Ij = maxJ[n][c];

        for (let i = 0; i < outH; i++) {
          const gRow = G.data[i];
          const iiRow = Ii.data[i];
          const ijRow = Ij.data[i];

          for (let j = 0; j < outW; j++) {
            const bi = iiRow[j] | 0;
            const bj = ijRow[j] | 0;
            dM.data[bi][bj] += gRow[j];
          }
        }

        outSample[c] = dM;
      }

      dX[n] = outSample;
    }

    this.#cache = null;
    return dX;
  }
}
