export class Vector {
  constructor(data) {
    this.data = data; // number[]
    this.size = data.length;
  }

  static fill(n, v) {
    return new Vector(Array(n).fill(v));
  }

  static zeros(n) {
    return this.fill(n, 0);
  }

  static fromJSON(json) {
    const source = typeof json === 'string' ? JSON.parse(json) : json;
    const result = Object.create(Vector.prototype);

    result.data = source;
    result.size = source.length;

    return result;
  }

  toJSON() {
    return this.data;
  }

  clone() {
    return new Vector([...this.data]);
  }

  map(fn) {
    return new Vector(this.data.map((x, i) => fn(x, i)));
  }

  sub(v) {
    return this.map((x, i) => x - v.data[i]);
  }

  scale(s) {
    return this.map((x) => x * s);
  }

  sum() {
    let s = 0;
    for (const x of this.data) {
      s += x;
    }
    return s;
  }

  mean() {
    return this.sum() / this.size;
  }
}
