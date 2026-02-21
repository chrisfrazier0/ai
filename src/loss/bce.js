bceLoss.typeTag = 'bce';

export function bceLoss(preds, targets, eps = 1e-12) {
  const B = preds.rowDim;
  const D = preds.colDim;
  const denom = B * D;

  // Clamp to avoid log(0) / div(0)
  // Could be avoided with logits optimization...
  const Y = preds.map((p) => Math.min(Math.max(p, eps), 1 - eps));

  // L = -mean( t*log(y) + (1-t)*log(1-y) )
  const loss = targets
    .mul(Y.map((y) => Math.log(y)))
    .add(targets.oneMinus().mul(Y.map((y) => Math.log(1 - y))))
    .scale(-1)
    .mean();

  // dL/dY = ((1-t) / (1-y) - t/y) / (B*D)
  const dY = targets
    .oneMinus()
    .mul(Y.map((y) => 1 / (1 - y)))
    .sub(targets.div(Y))
    .scale(1 / denom);

  return { loss, dY };
}
