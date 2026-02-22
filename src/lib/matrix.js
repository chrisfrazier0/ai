import { randn } from './math.js';
import { Vector } from './vector.js';

export class Matrix {
  constructor(rows) {
    this.data = rows; // number[][]
    this.rowDim = rows.length;
    this.colDim = rows[0].length;
  }

  static fill(r, c, v) {
    const rows = Array(r);
    for (let i = 0; i < r; i++) {
      rows[i] = Array(c).fill(v);
    }
    return new Matrix(rows);
  }

  static zeros(r, c) {
    return this.fill(r, c, 0);
  }

  static randn(r, c, std = 1) {
    const rows = [];
    for (let i = 0; i < r; i++) {
      const row = [];
      for (let j = 0; j < c; j++) {
        row.push(randn() * std);
      }
      rows.push(row);
    }
    return new Matrix(rows);
  }

  static oneHot(labels, numClasses) {
    return new Matrix(
      labels.map((k) => {
        const r = Array(numClasses).fill(0);
        const clamped = Math.min(Math.max(k, 0), numClasses - 1);
        r[clamped] = 1;
        return r;
      }),
    );
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = Object.create(Matrix.prototype);

    result.data = source;
    result.rowDim = source.length;
    result.colDim = source[0].length;

    return result;
  }

  toJSON() {
    return this.data;
  }

  clone() {
    return new Matrix(this.data.map((r) => [...r]));
  }

  map(fn) {
    return new Matrix(this.data.map((r, i) => r.map((x, j) => fn(x, i, j))));
  }

  add(M) {
    return this.map((x, i, j) => x + M.data[i][j]);
  }

  sub(M) {
    return this.map((x, i, j) => x - M.data[i][j]);
  }

  scale(s) {
    return this.map((x) => x * s);
  }

  mul(M) {
    return this.map((x, i, j) => x * M.data[i][j]);
  }

  matmul(M) {
    const out = [];
    for (let i = 0; i < this.rowDim; i++) {
      const row = [];
      for (let j = 0; j < M.colDim; j++) {
        let sum = 0;
        for (let k = 0; k < this.colDim; k++) {
          sum += this.data[i][k] * M.data[k][j];
        }
        row.push(sum);
      }
      out.push(row);
    }
    return new Matrix(out);
  }

  div(M) {
    return this.map((x, i, j) => x / M.data[i][j]);
  }

  transpose() {
    const out = [];
    for (let j = 0; j < this.colDim; j++) {
      const row = [];
      for (let i = 0; i < this.rowDim; i++) {
        row.push(this.data[i][j]);
      }
      out.push(row);
    }
    return new Matrix(out);
  }

  addScalar(s) {
    return this.map((x) => x + s);
  }

  addVector(v) {
    return this.map((x, _, j) => x + v.data[j]);
  }

  oneMinus() {
    return this.scale(-1).addScalar(1);
  }

  sumRows() {
    const sums = Array(this.colDim).fill(0);
    for (const r of this.data) {
      for (let j = 0; j < this.colDim; j++) {
        sums[j] += r[j];
      }
    }
    return new Vector(sums);
  }

  sumCols() {
    const sums = Array(this.rowDim);
    for (let i = 0; i < this.rowDim; i++) {
      let s = 0;
      for (let j = 0; j < this.colDim; j++) {
        s += this.data[i][j];
      }
      sums[i] = s;
    }
    return new Vector(sums);
  }

  takeRows(indices) {
    const rows = [];
    for (let i = 0; i < indices.length; i++) {
      rows.push([...this.data[indices[i]]]);
    }
    return new Matrix(rows);
  }

  sum() {
    let s = 0;
    for (const r of this.data) {
      for (const x of r) {
        s += x;
      }
    }
    return s;
  }

  mean() {
    return this.sum() / (this.rowDim * this.colDim);
  }

  rot180() {
    const rows = [];
    for (let i = this.rowDim - 1; i >= 0; i--) {
      rows.push([...this.data[i]].reverse());
    }
    return new Matrix(rows);
  }

  // Technically cross-correlation (no kernel flip), but conventionally
  // called convolution in deep learning since kernels are learned.
  // returns: (H-k+1) x (W-k+1)
  convolute(K) {
    const kH = K.rowDim;
    const kW = K.colDim;
    const outH = this.rowDim - kH + 1;
    const outW = this.colDim - kW + 1;

    const rows = Array(outH);
    for (let i = 0; i < outH; i++) {
      const row = new Array(outW);
      for (let j = 0; j < outW; j++) {
        let s = 0;
        for (let u = 0; u < kH; u++) {
          const xRow = this.data[i + u];
          const kRow = K.data[u];
          for (let v = 0; v < kW; v++) {
            s += xRow[j + v] * kRow[v];
          }
        }
        row[j] = s;
      }
      rows[i] = row;
    }
    return new Matrix(rows);
  }

  // returns: (H+kH-1) x (W+kW-1)
  convoluteFull(K) {
    return this.pad(K.rowDim - 1, K.colDim - 1).convolute(K);
  }

  pad(pH, pW = pH) {
    const P = Matrix.zeros(this.rowDim + 2 * pH, this.colDim + 2 * pW);
    for (let i = 0; i < this.rowDim; i++) {
      const src = this.data[i];
      const dst = P.data[i + pH];
      for (let j = 0; j < this.colDim; j++) {
        dst[j + pW] = src[j];
      }
    }
    return P;
  }

  unpad(pH, pW = pH) {
    const H = this.rowDim - 2 * pH;
    const W = this.colDim - 2 * pW;
    const rows = Array(H);
    for (let i = 0; i < H; i++) {
      rows[i] = this.data[i + pH].slice(pW, pW + W);
    }
    return new Matrix(rows);
  }
}
