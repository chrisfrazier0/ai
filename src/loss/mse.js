import { assert } from '../lib/assert.js';

mseLoss.typeTag = 'mse';

export function mseLoss(Y, T) {
  assert(
    Y.rowDim === T.rowDim && Y.colDim === T.colDim,
    `mseLoss: shape mismatch ${Y.rowDim}x${Y.colDim} vs ${T.rowDim}x${T.colDim}`,
  );

  const N = Y.rowDim;
  const D = Y.colDim;
  const diff = Y.sub(T);

  // L = (1/N*D) * sum (Y - T)^2
  const loss = diff
    .mul(diff)
    .scale(1 / (N * D))
    .sum();

  // dL/dY = 2*(Y - T)/(N*D)
  const dY = diff.scale(2 / (N * D));

  return { loss, dY };
}
