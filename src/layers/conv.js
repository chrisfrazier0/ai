import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';

export class ConvLayer {
  type = 'conv';
  #cache = null;

  constructor({ inChannels, outChannels, kernel = 3, std = 0.1, bias = 0 }) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernel = kernel;

    // W[f][c] = Matrix(kernel, kernel)
    this.W = Array.from({ length: outChannels }, () =>
      Array.from({ length: inChannels }, () =>
        Matrix.randn(kernel, kernel, std),
      ),
    );

    this.b = Array(outChannels).fill(bias);
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = new ConvLayer({ inChannels: 0, outChannels: 0 });
    Object.assign(result, source);

    for (let f = 0; f < result.W.length; f++) {
      for (let c = 0; c < result.W[f].length; c++) {
        result.W[f][c] = Matrix.fromJSON(result.W[f][c]);
      }
    }

    return result;
  }

  getState() {
    return {
      W: this.W.map((row) => row.map((m) => m.clone())),
      b: [...this.b],
    };
  }

  setState(state) {
    this.W = state.W.map((row) => row.map((m) => m.clone()));
    this.b = [...state.b];
  }

  forward(X) {
    const N = X.length;
    const C = this.inChannels;
    const F = this.outChannels;
    const k = this.kernel;

    assert(N > 0, 'ConvLayer.forward: empty batch');
    const H = X[0][0].rowDim;
    const W = X[0][0].colDim;

    // verify consistent shapes
    for (let n = 0; n < N; n++) {
      assert(
        X[n].length === C,
        `ConvLayer.forward: expected ${C} channels, got ${X[n].length} at sample ${n}`,
      );
      for (let c = 0; c < C; c++) {
        const M = X[n][c];
        assert(
          M.rowDim === H && M.colDim === W,
          `ConvLayer.forward: expected ${H}x${W}, got ${M.rowDim}x${M.colDim} at sample ${n} channel ${c}`,
        );
      }
    }

    const outH = H - k + 1;
    const outW = W - k + 1;
    assert(
      outH > 0 && outW > 0,
      `ConvLayer.forward: kernel ${k} too large for ${H}x${W}`,
    );

    const Y = new Array(N);
    for (let n = 0; n < N; n++) {
      const sample = X[n];
      const outSample = new Array(F);

      for (let f = 0; f < F; f++) {
        const Yf = Matrix.fill(outH, outW, this.b[f]);
        for (let c = 0; c < C; c++) {
          convolute$(Yf, sample[c], this.W[f][c]);
        }

        outSample[f] = Yf;
      }

      Y[n] = outSample;
    }

    this.#cache = { X, H, W };
    return Y;
  }

  backward(dY, lr) {
    assert(this.#cache, 'ConvLayer.backward: cache is null');
    const { X, H, W } = this.#cache;
    const N = X.length;
    const C = this.inChannels;
    const F = this.outChannels;
    const k = this.kernel;
    assert(
      dY.length === N,
      `ConvLayer.backward: expected batch length ${N}, got ${dY.length}`,
    );

    const outH = H - k + 1;
    const outW = W - k + 1;

    const dW = Array.from({ length: F }, () =>
      Array.from({ length: C }, () => Matrix.zeros(k, k)),
    );
    const db = Array(F).fill(0);
    const dX = Array.from({ length: N }, () =>
      Array.from({ length: C }, () => Matrix.zeros(H, W)),
    );

    for (let n = 0; n < N; n++) {
      const Xsample = X[n];
      const dXsample = dX[n];
      const dYsample = dY[n];
      assert(
        dYsample.length === F,
        `ConvLayer.backward: expected ${F} output channels, got ${dYsample.length} at sample ${n}`,
      );

      for (let f = 0; f < F; f++) {
        const G = dYsample[f];
        assert(
          G.rowDim === outH && G.colDim === outW,
          `ConvLayer.backward: expected ${outH}x${outW}, got ${G.rowDim}x${G.colDim} at sample ${n} outChannel ${f}`,
        );

        db[f] += G.sum();
        for (let c = 0; c < C; c++) {
          convoluteGradient$(dW[f][c], Xsample[c], G);
          convoluteInput$(dXsample[c], G, this.W[f][c]);
        }
      }
    }

    const invN = 1 / Math.max(N, 1);
    for (let f = 0; f < F; f++) {
      this.b[f] -= lr * (db[f] * invN);
      for (let c = 0; c < C; c++) {
        this.W[f][c] = this.W[f][c].sub(dW[f][c].scale(lr * invN));
      }
    }

    this.#cache = null;
    return dX;
  }
}

// X: H x W, K: k x k, out: (H-k+1) x (W-k+1)
function convolute$(out, X, K) {
  const H = X.rowDim;
  const W = X.colDim;
  const k = K.rowDim;

  const outH = H - k + 1;
  const outW = W - k + 1;

  for (let i = 0; i < outH; i++) {
    const outRow = out.data[i];

    for (let j = 0; j < outW; j++) {
      let s = 0;

      for (let u = 0; u < k; u++) {
        const xRow = X.data[i + u];
        const kRow = K.data[u];

        for (let v = 0; v < k; v++) {
          s += xRow[j + v] * kRow[v];
        }
      }

      outRow[j] += s;
    }
  }
}

// dK[u,v] += sum_{i,j} X[i+u, j+v] * dY[i,j]
function convoluteGradient$(dK, X, dY) {
  const k = dK.rowDim;
  const outH = dY.rowDim;
  const outW = dY.colDim;

  for (let u = 0; u < k; u++) {
    const dKrow = dK.data[u];

    for (let v = 0; v < k; v++) {
      let s = 0;

      for (let i = 0; i < outH; i++) {
        const xRow = X.data[i + u];
        const gRow = dY.data[i];

        for (let j = 0; j < outW; j++) {
          s += xRow[j + v] * gRow[j];
        }
      }

      dKrow[v] += s;
    }
  }
}

// dX = convolute(dY, rot180(K))
function convoluteInput$(dX, dY, K) {
  const H = dX.rowDim;
  const W = dX.colDim;
  const k = K.rowDim;

  const outH = dY.rowDim;
  const outW = dY.colDim;

  for (let i = 0; i < H; i++) {
    const dxRow = dX.data[i];

    for (let j = 0; j < W; j++) {
      let s = 0;

      // dY index (a,b) contributes to X(i,j) when (i-a) and (j-b) are within kernel bounds
      const a0 = Math.max(0, i - (k - 1));
      const a1 = Math.min(outH - 1, i);
      const b0 = Math.max(0, j - (k - 1));
      const b1 = Math.min(outW - 1, j);

      for (let a = a0; a <= a1; a++) {
        const gRow = dY.data[a];
        const u = i - a; // 0..k-1

        // rot180: row index flipped
        const kRow = K.data[k - 1 - u];

        for (let b = b0; b <= b1; b++) {
          const v = j - b; // 0..k-1

          // rot180: col index flipped
          s += gRow[b] * kRow[k - 1 - v];
        }
      }

      dxRow[j] += s;
    }
  }
}
