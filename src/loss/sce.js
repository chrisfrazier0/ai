sceLoss.typeTag = 'softmax_ce';

export function sceLoss(preds, targets, eps = 1e-12) {
  const B = preds.rowDim;
  const D = preds.colDim;

  // Clamp to avoid log(0) / div(0)
  // Could be avoided with logits optimization...
  const Y = preds.map((p) => Math.min(Math.max(p, eps), 1 - eps));

  // L = -mean(sum_col( t * log(y) ))
  const loss = targets
    .mul(Y.map((y) => Math.log(y)))
    .scale(-1)
    .sumCols()
    .mean();

  // dL/dY = -(t / y) / B
  const dY = targets.div(Y).scale(-1 / B);

  return { loss, dY };
}
