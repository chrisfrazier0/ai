import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { mlp } from '../src/factory.js';
import { assert } from '../src/lib/assert.js';
import { Matrix } from '../src/lib/matrix.js';

// ---- utility ----

function loadIdxImages(filePath) {
  const buf = fs.readFileSync(filePath);
  const magic = buf.readUInt32BE(0);
  assert(magic === 2051, `IDX images: bad magic ${magic}, expected 2051`);

  const n = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const size = rows * cols;

  const images = new Array(n);
  let p = 16;

  process.stdout.write(`Loading dataset (${n})... `);
  for (let i = 0; i < n; i++) {
    const row = new Array(size);
    for (let j = 0; j < size; j++) {
      row[j] = buf[p++]; // 0..255
    }
    images[i] = row;
  }

  console.log('Done!');
  return { images, n, rows, cols };
}

function loadIdxLabels(filePath) {
  const buf = fs.readFileSync(filePath);
  const magic = buf.readUint32BE(0);
  assert(magic === 2049, `IDX labels: bad magic ${magic}, expected 2049`);

  const n = buf.readUint32BE(4);
  const labels = new Array(n);
  let p = 8;

  process.stdout.write(`Loading labels (${n})... `);
  for (let i = 0; i < n; i++) {
    labels[i] = buf[p++]; // 0..9
  }

  console.log('Done!');
  return { labels, n };
}

function pseudoNormalize(images) {
  return new Matrix(images.map((row) => row.map((px) => (px / 255) * 2 - 1)));
}

function accuracy(model, probsMatrix, labels) {
  assert(
    probsMatrix.rowDim === labels.length,
    `accuracy: expected ${labels.length} rows, got ${probsMatrix.rowDim}`,
  );

  let correct = 0;
  for (let i = 0; i < labels.length; i++) {
    const pred = model.finalize(probsMatrix.data[i]);
    if (pred === labels[i]) correct++;
  }
  return correct / labels.length;
}

// ---- main ----

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const trainImagesPath = path.join(__dirname, '../data/train-images-idx3-ubyte');
const trainLabelsPath = path.join(__dirname, '../data/train-labels-idx1-ubyte');
const testImagesPath = path.join(__dirname, '../data/t10k-images-idx3-ubyte');
const testLabelsPath = path.join(__dirname, '../data/t10k-labels-idx1-ubyte');
const outPath = path.join(__dirname, '../data/model_mlp.json');

const trainImgs = loadIdxImages(trainImagesPath);
const trainLabs = loadIdxLabels(trainLabelsPath);
const testImgs = loadIdxImages(testImagesPath);
const testLabs = loadIdxLabels(testLabelsPath);

assert(
  trainImgs.n === trainLabs.n,
  `train mismatch: ${trainImgs.n} images vs ${trainLabs.n} labels`,
);
assert(
  testImgs.n === testLabs.n,
  `test mismatch: ${testImgs.n} images vs ${testLabs.n} labels`,
);

const nTrain = 60_000;
const Xtrain = pseudoNormalize(trainImgs.images.slice(0, nTrain));
const Ttrain = Matrix.oneHot(trainLabs.labels.slice(0, nTrain), 10);

const model = mlp({
  activation: 'relu',
  leaky: 0.01,
  final: 'softmax',
  loss: 'sce',
  layers: [784, 256, 10],
  batchSize: 64,
  learningRate: 0.05,
  epochs: 30,
  minImprovement: 1e-2,
  patience: 7,
});

console.log('training...');
const result = model.train(Xtrain, Ttrain, 0.1, true);
console.log(result);

console.log('evaluating...');
const Xtest = pseudoNormalize(testImgs.images);
const Ptest = model.forward(Xtest);
const acc = accuracy(model, Ptest, testLabs.labels);
console.log('accuracy:', (acc * 100).toFixed(3) + '%');

const json = JSON.stringify(model);
fs.writeFileSync(outPath, json, 'utf8');
console.log('saved model:', outPath);
