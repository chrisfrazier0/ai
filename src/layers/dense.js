import { assert } from '../lib/assert.js';
import { Matrix } from '../lib/matrix.js';
import { Vector } from '../lib/vector.js';

export class DenseLayer {
  type = 'dense';
  #cache = null;

  constructor({ weights, bias = 0 }) {
    this.W = weights;
    this.b = Vector.fill(weights?.colDim ?? 0, bias);
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = new DenseLayer({ weights: null });
    Object.assign(result, source);

    result.W = Matrix.fromJSON(result.W);
    result.b = Vector.fromJSON(result.b);

    return result;
  }

  getState() {
    return {
      W: this.W.clone(),
      b: this.b.clone(),
    };
  }

  setState(state) {
    this.W = state.W.clone();
    this.b = state.b.clone();
  }

  forward(X) {
    assert(
      X.colDim === this.W.rowDim,
      `DenseLayer.forward: expected ${this.W.rowDim}, got ${X.colDim}`,
    );

    // Y = XW + b
    const Y = X.matmul(this.W).addVector(this.b);
    this.#cache = { X };
    return Y;
  }

  backward(dY, lr) {
    assert(this.#cache, 'DenseLayer.backward: cache is null');
    const { X } = this.#cache;
    assert(
      dY.rowDim === X.rowDim,
      `DenseLayer.backward: expected dY.rowDim ${X.rowDim}, got ${dY.rowDim}`,
    );
    assert(
      dY.colDim === this.W.colDim,
      `DenseLayer.backward: expected dY.colDim ${this.W.colDim}, got ${dY.colDim}`,
    );

    // dL/dW = X^T dY
    const dW = X.transpose().matmul(dY);

    // dL/db = dY
    const db = dY.sumRows();

    // dL/dX = dY W^T
    const dX = dY.matmul(this.W.transpose());

    this.W = this.W.sub(dW.scale(lr));
    this.b = this.b.sub(db.scale(lr));
    this.#cache = null;
    return dX;
  }
}
