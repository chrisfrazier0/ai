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

    // Y[n][f] = b[f] + sum_c convolute(X[n][c], W[f][c])
    const Y = new Array(N);
    for (let n = 0; n < N; n++) {
      const outSample = new Array(F);
      for (let f = 0; f < F; f++) {
        let Yf = Matrix.fill(outH, outW, this.b[f]);
        for (let c = 0; c < C; c++) {
          Yf = Yf.add(X[n][c].convolute(this.W[f][c]));
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

        // dL/db = sum of gradient
        db[f] += G.sum();

        for (let c = 0; c < C; c++) {
          // dL/dW = convolute(X, G)
          dW[f][c] = dW[f][c].add(X[n][c].convolute(G));

          // dL/dX = convoluteFull(G, rot180(W))
          dX[n][c] = dX[n][c].add(G.convoluteFull(this.W[f][c].rot180()));
        }
      }
    }

    for (let f = 0; f < F; f++) {
      this.b[f] -= db[f] * lr;
      for (let c = 0; c < C; c++) {
        this.W[f][c] = this.W[f][c].sub(dW[f][c].scale(lr));
      }
    }

    this.#cache = null;
    return dX;
  }
}
