import { mlp } from '../src/factory.js';
import { assert } from '../src/lib/assert.js';
import { Matrix } from '../src/lib/matrix.js';

const X = new Matrix([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const T = new Matrix([[0], [1], [1], [0]]);

const model = mlp({
  activation: 'sigmoid',
  final: 'sigmoid',
  loss: 'bce',
  layers: [2, 3, 1],
  batchSize: 1,
  learningRate: 1,
  epochs: 20_000,
  minImprovement: 1e-1,
  patience: 700,
});

const Xn = X.map((x) => 2 * x - 1); // pseudo normalize
const result = model.train(Xn, T);
const P = model.forward(Xn);
const final = model.finalize(P);

console.log('train:', result);
console.log(
  'probs:',
  P.data.map((r) => r[0]),
);

assert(final.data[0][0] === 0, 'XOR failed: (0,0) should be 0');
assert(final.data[1][0] === 1, 'XOR failed: (0,1) should be 1');
assert(final.data[2][0] === 1, 'XOR failed: (1,0) should be 1');
assert(final.data[3][0] === 0, 'XOR failed: (1,1) should be 0');

console.log('XOR tests passed');
