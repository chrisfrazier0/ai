// Gaussian random (Box–Muller)
export function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function softplus(z) {
  return Math.max(z, 0) + Math.log(1 + Math.exp(-Math.abs(z)));
}

export function sigmoid(z) {
  return Math.exp(-softplus(-z));
}

export function tanh(z) {
  return 2 * sigmoid(2 * z) - 1;
}
