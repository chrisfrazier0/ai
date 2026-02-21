import { Matrix } from '../lib/matrix.js';

sceLoss.typeTag = 'softmax_ce';

export function sceLoss(preds, targets, eps = 1e-12) {
  const B = preds.rowDim;
  const D = preds.colDim;

  // Clamp to avoid log(0) / div(0)
  // Could be avoided with logits optimization...
  const Y = preds.map((p) => Math.min(Math.max(p, eps), 1 - eps));

  // L = -mean(sum( t * log(y) ))
  let lossSum = 0;
  for (let i = 0; i < B; i++) {
    let rowSum = 0;
    for (let j = 0; j < D; j++) {
      rowSum += targets.data[i][j] * Math.log(Y.data[i][j]);
    }
    lossSum += rowSum;
  }
  const loss = -lossSum / B;

  // dL/dY = -(t / y) / B
  const dY = new Matrix(
    targets.data.map((row, i) => {
      const out = new Array(D);
      for (let j = 0; j < D; j++) {
        out[j] = -(row[j] / Y.data[i][j]) / B;
      }
      return out;
    }),
  );

  return { loss, dY };
}
