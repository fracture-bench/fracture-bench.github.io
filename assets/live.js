// Playground "live" mode: the surrogates run in the visitor's browser on a new case.
//
// Dataset 1: DeepONet and Latent DeepONet. Dataset 2: FNO, DeepONet and Latent
// DeepONet. (The Dataset 1 FNO, 1.2 GB per model, is too large for a browser.)
// The networks are the ones in models/_flax_arch.py and models/fno.py of the
// Streamlit version, rebuilt with TensorFlow.js. Weights, grids and constants
// come from live/, written by tools/export_live_models.py.
//
// Shortcuts, exact up to rounding:
// * DeepONet: the trunk net depends only on the grid and its last layer is
//   linear, so the branch-trunk product folds into one short vector:
//       sum_p b_p (H W + c)_p  =  H (W b) + c.b
//   which skips the widest layer (512 x p) at every grid point.
// * Latent DeepONet, Dataset 1: the same idea is applied when exporting; the
//   operator becomes latent = h M + B.
// * FNO: only 32 x 48 Fourier modes are kept, so the FFTs become small
//   matrix products with cosine and sine tables.

(() => {
  "use strict";

  const TF_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
  const DIR = "live/";
  const CHUNK = 16384;      // DeepONet trunk points per GPU pass
  const AUTO_MS = 2500;     // re-run on every change when a prediction takes less than this
  const NAMES = { fno: "FNO", deeponet: "DeepONet", latent_deeponet: "Latent DeepONet" };

  // ------------------------------------------------------------ downloads
  const memo = new Map();
  function once(key, make) {
    if (!memo.has(key)) memo.set(key, make().catch((e) => { memo.delete(key); throw e; }));
    return memo.get(key);
  }

  const getConfig = () => once("config", async () => {
    const r = await fetch(DIR + "live.json");
    if (!r.ok) throw new Error(`live.json: HTTP ${r.status}`);
    return r.json();
  });

  const getTF = () => once("tf", () => new Promise((resolve, reject) => {
    if (window.tf) return resolve(window.tf);
    const s = document.createElement("script");
    s.src = TF_URL;
    s.onload = () => window.tf.ready().then(() => {
      // pass shapes as uniforms so GPU programs are compiled once, not per tensor shape
      // (cuts the first prediction from ~11 s to ~4 s)
      try { window.tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true); } catch (e) { /* not WebGL */ }
      resolve(window.tf);
    }, reject);
    s.onerror = () => { s.remove(); reject(new Error("TensorFlow.js could not be downloaded")); };
    document.head.append(s);
  }));

  async function download(url, onBytes) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    if (!r.body) return r.arrayBuffer();
    const reader = r.body.getReader();
    const parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      got += value.length;
      if (onBytes) onBytes(got);
    }
    const buf = new Uint8Array(got);
    let at = 0;
    for (const p of parts) { buf.set(p, at); at += p.length; }
    return buf.buffer;
  }

  let HALF = null;                      // float16 -> float32 lookup table
  function halfToFloat(u16) {
    if (!HALF) {
      HALF = new Float32Array(65536);
      for (let h = 0; h < 65536; h++) {
        const sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
        HALF[h] = sign * (e === 0 ? m * 2 ** -24
          : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
      }
    }
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = HALF[u16[i]];
    return out;
  }

  const product = (shape) => shape.reduce((a, b) => a * b, 1);
  const mb = (bytes) => `${Math.round(bytes / 1e6)} MB`;

  // A model's weights: one or more files, read as one byte stream. Each tensor
  // is [byte offset, shape, "f32" | "f16"]; returns {name: tf.Tensor}.
  function getWeights(spec, onFraction) {
    return once("w:" + spec.files[0], async () => {
      await getTF();
      const all = new Uint8Array(spec.bytes);
      let done = 0;
      for (const f of spec.files) {
        const buf = new Uint8Array(await download(DIR + f, (got) => {
          if (onFraction) onFraction(Math.min((done + got) / spec.bytes, 1));
        }));
        all.set(buf, done);
        done += buf.length;
      }
      const W = {};
      for (const [name, [off, shape, dtype]] of Object.entries(spec.params)) {
        const n = product(shape);
        const data = dtype === "f16"
          ? halfToFloat(new Uint16Array(all.buffer, off, n))
          : new Float32Array(all.buffer, off, n);
        W[name] = tf.tensor(data, shape, "float32");
      }
      return W;
    });
  }

  function backendNote() {
    const b = tf.getBackend();
    if (b === "webgl") {
      let full = true;
      try { full = tf.env().getBool("WEBGL_RENDER_FLOAT32_CAPABLE"); } catch (e) { /* older flag set */ }
      return "your graphics card (WebGL)" + (full ? "" : ", at reduced precision on this device");
    }
    return b === "cpu" ? "the CPU, which is slow: try a browser with WebGL" : b;
  }

  // ------------------------------------------------------------ network layers
  // Call these inside tf.tidy so the intermediate tensors are freed.
  const GELU_C = Math.sqrt(2 / Math.PI);
  const gelu = (x) =>                   // tanh approximation, as flax.linen.gelu
    x.mul(tf.tanh(x.add(x.mul(x).mul(x).mul(0.044715)).mul(GELU_C)).add(1)).mul(0.5);
  const swish = (x) => x.mul(tf.sigmoid(x));             // = flax silu / swish
  const dense = (W, name, x) => x.matMul(W[name + "/kernel"]).add(W[name + "/bias"]);
  const conv = (W, name, x, stride) =>
    tf.conv2d(x, W[name + "/kernel"], stride, "same").add(W[name + "/bias"]);

  function trunkHidden(W, x, layers) {  // DeepONet trunk's hidden layers, 512 features out
    for (let i = 0; i < layers; i++) x = gelu(dense(W, `trunk_net/Dense_${i}`, x));
    return x;
  }

  // Fold DeepONet's last trunk layer into the branch output (see the note at the top).
  // b: [1, p] for `ch` output channels  ->  v: [512, ch], c: [ch]
  function fold(W, last, b, ch) {
    const K = W[last + "/kernel"];
    const P = b.shape[1] / ch;
    const bb = b.reshape([ch, P]);
    const v = K.reshape([K.shape[0], ch, P]).mul(bb.expandDims(0)).sum(2);
    const c = W[last + "/bias"].reshape([ch, P]).mul(bb).sum(1);
    return [v, c];
  }

  // Dataset 1 DeepONet branch: CNN on (128, 128, 4) = [u_x, u_y (normalized), grid_x, grid_y]
  function branchD1(W, x) {
    const n = (s) => "branch_net/" + s;
    let h = gelu(conv(W, n("Conv_0"), x, 1));
    h = tf.maxPool(h, 2, 2, "same");
    h = gelu(conv(W, n("Conv_1"), h, 1));
    h = gelu(conv(W, n("Conv_2"), h, 2));
    h = gelu(conv(W, n("Conv_3"), h, 2));
    h = tf.maxPool(h, 2, 2, "same");
    h = gelu(conv(W, n("Conv_4"), h, 2));
    h = h.mean([1, 2]);
    h = gelu(dense(W, n("Dense_0"), h));
    return dense(W, n("Dense_1"), h);
  }

  // Dataset 1 Latent DeepONet branch, up to its 512 hidden features
  function branchD1Latent(W, x) {
    let h = x;
    [1, 2, 2, 2].forEach((s, i) => { h = swish(conv(W, `branch_net/Conv_${i}`, h, s)); });
    return swish(dense(W, "branch_net/Dense_0", h.mean([1, 2])));
  }

  // Dataset 2 branch (DeepONet and Latent DeepONet): residual CNN on (128, 256, 2)
  function branchD2(W, x) {
    const n = (i) => `branch_net/Conv_${i}`;
    const block = (h, i, stride) => {
      let y = swish(conv(W, n(i), h, stride));
      y = conv(W, n(i + 1), y, 1);
      return swish(y.add(conv(W, n(i + 2), h, stride)));   // 1x1 shortcut
    };
    let h = block(x, 0, 1);
    h = block(h, 3, [1, 2]);
    h = block(h, 6, 2);
    h = block(h, 9, 2);
    h = tf.avgPool(h, 8, 8, "valid").reshape([1, -1]);
    h = gelu(dense(W, "branch_net/Dense_0", h));
    h = gelu(dense(W, "branch_net/Dense_1", h));
    return dense(W, "branch_net/Dense_2", h);
  }

  // Latent DeepONet decoder: 4 dense layers, leaky ReLU (slope 0.01) between them
  function decode(W, z) {
    for (let i = 0; i < 4; i++) {
      z = dense(W, `decoder/layers_${i}`, z);
      if (i < 3) z = tf.leakyRelu(z, 0.01);
    }
    return z;
  }

  // ------------------------------------------------------------ Dataset 1 models
  const linspace = (a, b, n) => Float64Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

  function holeMask(d1) {               // true inside the hole, (row, col) layout
    const G = d1.grid[2], s = linspace(d1.grid[0], d1.grid[1], G), m = new Uint8Array(G * G);
    for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) m[a * G + b] = s[a] ** 2 + s[b] ** 2 < d1.hole_r2;
    return m;
  }

  // branch input (128, 128, 4) = [u_x, u_y (normalized, zero in the hole), grid_x, grid_y]
  function d1Input(d1, hole, u0) {
    const G = d1.grid[2], s = linspace(d1.grid[0], d1.grid[1], G);
    const inp = new Float32Array(G * G * 4);
    for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) {
      const k = a * G + b, keep = hole[k] ? 0 : 1;
      for (let c = 0; c < 2; c++)
        inp[k * 4 + c] = (u0[k * 2 + c] * d1.disp_scale * keep - d1.disp_min[c]) / (d1.disp_max[c] - d1.disp_min[c]);
      inp[k * 4 + 2] = s[b];
      inp[k * 4 + 3] = s[a];
    }
    return tf.tensor4d(inp, [1, G, G, 4]);
  }

  // Each returns the vector phase field (n, 128, 128, 2), zero inside the hole.
  async function deeponetD1(W, d1, hole, u0, n) {
    const G = d1.grid[2], s = linspace(d1.grid[0], d1.grid[1], G);
    const [v, c] = tf.tidy(() => fold(W, "trunk_net/Dense_4", branchD1(W, d1Input(d1, hole, u0)), 2));
    // trunk points (t, x, y): t over the n load steps, y varying fastest
    const t = linspace(0, 1, n), N = n * G * G, out = new Float32Array(N * 2);
    try {
      for (let start = 0; start < N; start += CHUNK) {
        const end = Math.min(N, start + CHUNK), pts = new Float32Array((end - start) * 3);
        for (let k = start; k < end; k++) {
          const o = (k - start) * 3;
          pts[o] = t[Math.floor(k / (G * G))];
          pts[o + 1] = s[Math.floor(k / G) % G];
          pts[o + 2] = s[k % G];
        }
        const y = tf.tidy(() =>
          tf.tanh(trunkHidden(W, tf.tensor2d(pts, [end - start, 3]), 4).matMul(v).add(c)));
        out.set(await y.data(), start * 2);
        y.dispose();
      }
    } finally {
      v.dispose();
      c.dispose();
    }
    for (let k = 0; k < N; k++) if (hole[k % (G * G)]) out[2 * k] = out[2 * k + 1] = 0;
    return out;
  }

  async function latentD1(W, d1, hole, u0, n) {
    const G = d1.grid[2], L = 256;
    const recon = tf.tidy(() => {
      const h = branchD1Latent(W, d1Input(d1, hole, u0));                      // [1, 512]
      const z = h.matMul(W["fold/M"]).add(W["fold/B"]).reshape([n, L, 2]);    // latent (t, l, i)
      return decode(W, z.transpose([0, 2, 1]).reshape([2 * n, L]));           // row t*2+i
    });
    const flat = await recon.data();
    recon.dispose();
    // the training code unpacked the decoded rows with a raw reshape to (16384, n, 2)
    const out = new Float32Array(n * G * G * 2);
    for (let a = 0; a < G * G; a++) {
      if (hole[a]) continue;
      for (let t = 0; t < n; t++) for (let i = 0; i < 2; i++)
        out[(t * G * G + a) * 2 + i] = flat[a * 2 * n + t * 2 + i];
    }
    return out;
  }

  // Boundary profile alpha(s) from the periodic Gaussian random field of the FEM
  // data generator (utils/elasticity.spectral_boundary_profile).
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normals(seed, count) {
    const rand = mulberry32(seed), z = [];
    while (z.length < count) {
      const u = rand() || 1e-12, w = rand(), r = Math.sqrt(-2 * Math.log(u));
      z.push(r * Math.cos(2 * Math.PI * w), r * Math.sin(2 * Math.PI * w));
    }
    return z;
  }

  function boundaryProfile(grf, G, seed, p) {
    const M = grf.n_modes, z = normals(seed, 2 * M);
    const lam = Array.from({ length: M }, (_, i) =>
      Math.SQRT2 * p.sigma * ((2 * Math.PI * (i + 1)) ** 2 + p.tau ** 2) ** (-p.gamma / 2));
    const alpha = new Float64Array(G);
    for (let k = 0; k < G; k++) {
      const phi = (2 * Math.PI * k) / (G - 1);      // 2 pi (s - a) / (b - a)
      let f = p.mean;
      for (let i = 0; i < M; i++)
        f += lam[i] * (z[i] * Math.cos((i + 1) * phi) + z[M + i] * Math.sin((i + 1) * phi));
      alpha[k] = grf.u_r * grf.t_first * f;
    }
    return alpha;
  }

  // u(x, 0) = sum_j alpha_j basis_j: the first load step is linear elastic.
  function displacement(basis, alpha) {
    const M = basis.length / alpha.length, u = new Float64Array(M);
    for (let j = 0; j < alpha.length; j++) {
      const a = alpha[j], o = j * M;
      for (let k = 0; k < M; k++) u[k] += a * basis[o + k];
    }
    return u;
  }

  // ------------------------------------------------------------ Dataset 2 models
  // Inputs d0 and en (E normalized with the training min/max) are (256, 128) = (y, x);
  // the networks use (x, y) = (128, 256). Each model returns d(x, t_f) as (y, x).
  function toYX(flat, H, X) {
    const out = new Float32Array(H * X);
    for (let x = 0; x < X; x++) for (let j = 0; j < H; j++)
      out[j * X + x] = Math.min(1, Math.max(0, flat[x * H + j]));
    return out;
  }

  function d2Input(H, X, d0, en) {      // (1, 128, 256, 2) = [d(x,0), E]
    const inp = new Float32Array(X * H * 2);
    for (let x = 0; x < X; x++) for (let y = 0; y < H; y++) {
      inp[(x * H + y) * 2] = d0[y * X + x];
      inp[(x * H + y) * 2 + 1] = en[y * X + x];
    }
    return tf.tensor4d(inp, [1, X, H, 2]);
  }

  let trunkD2 = null;                   // DeepONet trunk features on the grid, kept on the GPU (64 MB)

  async function deeponetD2(W, spec, d2, d0, en) {
    const [H, X] = d2.shape;
    if (!trunkD2) {                     // physical coordinates in mm, y varying fastest
      const xs = linspace(0, 4, X), ys = linspace(0, 8, H), pts = new Float32Array(X * H * 2);
      for (let i = 0; i < X; i++) for (let j = 0; j < H; j++) {
        pts[(i * H + j) * 2] = xs[i];
        pts[(i * H + j) * 2 + 1] = ys[j];
      }
      trunkD2 = tf.tidy(() => trunkHidden(W, tf.tensor2d(pts, [X * H, 2]), 3));
    }
    const y = tf.tidy(() => {
      const [v, c] = fold(W, "trunk_net/Dense_3", branchD2(W, d2Input(H, X, d0, en)), 1);
      return trunkD2.matMul(v).add(c);
    });
    const flat = await y.data();
    y.dispose();
    return toYX(flat, H, X);
  }

  async function latentD2(W, spec, d2, d0, en) {
    const [H, X] = d2.shape;
    const y = tf.tidy(() => decode(W, branchD2(W, d2Input(H, X, d0, en))));
    const flat = await y.data();
    y.dispose();
    return toYX(flat, H, X);
  }

  // FNO: lift to 64 channels, three Fourier layers, project to one channel.
  // Activations are kept as [H*W, C] with H = 128 (x) and W = 256 (y).
  const dft = new Map();
  function dftTables(H, Wd, m1, m2) {   // cosine / sine tables for the kept modes
    const key = `${H}x${Wd}x${m1}x${m2}`;
    if (!dft.has(key)) {
      const table = (rows, cols, f) => {
        const a = new Float32Array(rows * cols);
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) a[r * cols + c] = f(r, c);
        return tf.tensor2d(a, [rows, cols]);
      };
      const ang = (k, n, N) => (2 * Math.PI * ((k * n) % N)) / N;
      const wt = (k) => (k === 0 ? 1 : 2) / Wd;       // inverse real DFT weights
      dft.set(key, {
        cw: table(Wd, m2, (w, k) => Math.cos(ang(k, w, Wd))),
        sw: table(Wd, m2, (w, k) => Math.sin(ang(k, w, Wd))),
        ch: table(m1, H, (k, h) => Math.cos(ang(k, h, H))),
        sh: table(m1, H, (k, h) => Math.sin(ang(k, h, H))),
        chT: table(H, m1, (h, k) => Math.cos(ang(k, h, H)) / H),
        shT: table(H, m1, (h, k) => Math.sin(ang(k, h, H)) / H),
        ci: table(m2, Wd, (k, w) => wt(k) * Math.cos(ang(k, w, Wd))),
        si: table(m2, Wd, (k, w) => wt(k) * Math.sin(ang(k, w, Wd))),
      });
    }
    return dft.get(key);
  }

  // irfft2(rfft2(a)[:, :, :m1, :m2] mixed by the layer weights), as in models/fno.py
  function spectral(W, i, a, H, Wd, C, m1, m2, T) {
    const A = a.reshape([H, Wd, C]).transpose([0, 2, 1]).reshape([H * C, Wd]);
    const Fr = A.matMul(T.cw).reshape([H, C * m2]);            // along y: e^{-i k w}
    const Fi = A.matMul(T.sw).neg().reshape([H, C * m2]);
    const Gr = T.ch.matMul(Fr).add(T.sh.matMul(Fi));            // along x: e^{-i k h}
    const Gi = T.ch.matMul(Fi).sub(T.sh.matMul(Fr));
    const modes = (g) => g.reshape([m1, C, m2]).transpose([0, 2, 1]).reshape([m1 * m2, 1, C]);
    const Xr = modes(Gr), Xi = modes(Gi);
    const wr = W[`SpectralConv2d_${i}/wr`], wi = W[`SpectralConv2d_${i}/wi`];   // [modes, C, C]
    const Yr = tf.matMul(Xr, wr).sub(tf.matMul(Xi, wi)).reshape([m1, m2 * C]);
    const Yi = tf.matMul(Xr, wi).add(tf.matMul(Xi, wr)).reshape([m1, m2 * C]);
    const Zr = T.chT.matMul(Yr).sub(T.shT.matMul(Yi));          // inverse along x
    const Zi = T.chT.matMul(Yi).add(T.shT.matMul(Yr));
    const back = (z) => z.reshape([H, m2, C]).transpose([0, 2, 1]).reshape([H * C, m2]);
    const y = back(Zr).matMul(T.ci).sub(back(Zi).matMul(T.si));  // inverse real DFT along y
    return y.reshape([H, C, Wd]).transpose([0, 2, 1]).reshape([H * Wd, C]);
  }

  async function fnoD2(W, spec, d2, d0, en) {
    const [H2, X2] = d2.shape, H = X2, Wd = H2, C = 64, [m1, m2] = spec.modes;
    // input channels [d(x,0), E min-maxed per sample, grid_x in [0,1], grid_y in [0,2]], then a*x + b
    let lo = Infinity, hi = -Infinity;
    for (const v of en) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const gx = linspace(0, 1, H), gy = linspace(0, 2, Wd), inp = new Float32Array(H * Wd * 4);
    for (let i = 0; i < H; i++) for (let j = 0; j < Wd; j++) {
      const o = (i * Wd + j) * 4, k = j * X2 + i;
      inp[o] = spec.a * d0[k] + spec.b;
      inp[o + 1] = spec.a * (hi > lo ? (en[k] - lo) / (hi - lo) : 0) + spec.b;
      inp[o + 2] = spec.a * gx[i] + spec.b;
      inp[o + 3] = spec.a * gy[j] + spec.b;
    }
    const T = dftTables(H, Wd, m1, m2);
    const y = tf.tidy(() => {
      let a = gelu(dense(W, "Dense_0", tf.tensor2d(inp, [H * Wd, 4])));
      for (let i = 0; i < 3; i++) {
        a = spectral(W, i, a, H, Wd, C, m1, m2, T).add(a.matMul(W[`SimpleConv2d_${i}/kernel`]));
        if (i < 2) a = gelu(a);
      }
      a = gelu(dense(W, "Dense_1", a));
      a = gelu(dense(W, "Dense_2", a));
      return dense(W, "Dense_3", a);
    });
    const flat = await y.data();
    y.dispose();
    return toYX(flat, Wd, H);
  }

  const inDisc = (x, y, D, extra = 0) => (x - D.col) ** 2 + (y - D.row) ** 2 < (D.r + extra) ** 2;

  // PF-CZM pre-crack d(x, 0) from the left edge to the tip (utils/geometry.build_initial_crack)
  function initialCrack(d2, tx, ty) {
    const [H, X] = d2.shape, { half_width: hw, tip_decay: td, clearance: cl } = d2.crack;
    const d0 = new Float32Array(H * X);
    for (let y = 0; y < H; y++) for (let x = 0; x < X; x++) {
      if (inDisc(x, y, d2.upper, cl) || inDisc(x, y, d2.lower, cl)) continue;
      const across = Math.exp(-((y - ty) ** 2) / (2 * hw * hw));
      const along = x <= tx ? 1 : Math.exp(-(((x - tx) / td) ** 2));
      d0[y * X + x] = Math.min(1, Math.max(0, across * along));
    }
    return d0;
  }

  // ------------------------------------------------------------ drawing
  function palette(hexes) {
    const p = new Uint8Array(hexes.length * 3);
    hexes.forEach((h, i) => {
      const n = parseInt(h.slice(1), 16);
      p.set([n >> 16, (n >> 8) & 255, n & 255], i * 3);
    });
    return p;
  }

  const level = (v, vmax) => Math.round(Math.min(1, Math.max(0, v / vmax)) * 254);

  // index: palette entries in (row, col) layout; row 0 is drawn at the bottom
  function paint(canvas, rows, cols, index, pal) {
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = index[(rows - 1 - r) * cols + c] * 3, o = (r * cols + c) * 4;
      img.data[o] = pal[i];
      img.data[o + 1] = pal[i + 1];
      img.data[o + 2] = pal[i + 2];
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function colorbar(el, hexes, label) {
    const stops = Array.from({ length: 11 }, (_, i) => hexes[Math.round((i * 254) / 10)]);
    el.innerHTML = `<span>0</span><div class="grad" style="background:linear-gradient(to right, ${stops.join(", ")})"></div><span>1</span><em>${label}</em>`;
  }

  function profileChart(svg, alpha) {
    const W = 300, H = 112, L = 34, R = 8, T = 8, B = 24;
    const mean = alpha.reduce((a, b) => a + b, 0) / alpha.length;
    const f = Array.from(alpha, (a) => a / Math.max(Math.abs(mean), 1e-12));
    const lo = Math.min(...f), hi = Math.max(...f), span = hi - lo || 1;
    const px = (i) => L + ((W - L - R) * i) / (f.length - 1);
    const py = (v) => T + (H - T - B) * (1 - (v - lo) / span);
    const line = f.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    svg.innerHTML = `
      <polygon points="${px(0)},${py(lo)} ${line} ${px(f.length - 1)},${py(lo)}" fill="#245E94" fill-opacity=".18"/>
      <polyline points="${line}" fill="none" stroke="#245E94" stroke-width="1.8"/>
      <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#D8DFEA"/>
      <text x="${L - 5}" y="${py(hi) + 4}" text-anchor="end">${hi.toFixed(2)}</text>
      <text x="${L - 5}" y="${py(lo) + 4}" text-anchor="end">${lo.toFixed(2)}</text>
      <text x="${L}" y="${H - 7}">−1</text>
      <text x="${W - R}" y="${H - 7}" text-anchor="end">1</text>
      <text x="${(L + W - R) / 2}" y="${H - 7}" text-anchor="middle">boundary coordinate s</text>`;
  }

  const seconds = (ms) => (ms < 1000 ? `${Math.max(1, Math.round(ms))} ms` : `${(ms / 1000).toFixed(1)} s`);

  // Model checkboxes; label(key) gives the text after the name (e.g. the download size).
  function modelPicker(container, keys, checked, label, onChange) {
    container.innerHTML = "";
    const spans = {};
    keys.forEach((k) => {
      const lab = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = k;
      box.checked = checked.includes(k);
      box.addEventListener("change", onChange);
      spans[k] = document.createElement("small");
      lab.append(box, document.createTextNode(NAMES[k] + " "), spans[k]);
      container.append(lab);
    });
    const refresh = () => keys.forEach((k) => { spans[k].textContent = label(k); });
    refresh();
    return {
      chosen: () => [...container.querySelectorAll("input:checked")].map((b) => b.value),
      refresh,
    };
  }

  // One canvas per model, with a caption.
  function makePanels(container, keys) {
    container.innerHTML = "";
    const panels = {};
    keys.forEach((k) => {
      const fig = document.createElement("figure");
      fig.className = "panel";
      fig.innerHTML = '<canvas role="img"></canvas><figcaption></figcaption>';
      fig.firstChild.setAttribute("aria-label", `${NAMES[k]} prediction`);
      container.append(fig);
      panels[k] = { canvas: fig.firstChild, caption: fig.lastChild };
    });
    return panels;
  }

  // Shared run logic: one run at a time, re-run once if inputs changed meanwhile.
  // work(say) runs the chosen models and returns the total compute time in ms.
  function runner(button, status, frame, work) {
    let busy = false, again = false, lastMs = Infinity, hasResult = false;
    const say = (text, error) => {
      status.textContent = text;
      status.classList.toggle("error", Boolean(error));
    };
    async function run() {
      if (busy) { again = true; return; }
      busy = true;
      button.disabled = true;
      button.classList.remove("attention");
      try {
        const ms = await work(say);
        if (ms === null) { say("Select at least one surrogate."); return; }
        lastMs = ms;
        hasResult = true;
        frame.classList.remove("stale");
        say(`Predicted in ${seconds(ms)} on ${backendNote()}.`);
      } catch (e) {
        console.error(e);
        say(`The model could not run in this browser: ${e.message}`, true);
      } finally {
        busy = false;
        button.disabled = false;
        if (again) { again = false; run(); }
      }
    }
    function changed() {                // call when an input changes
      if (!hasResult) return;
      if (lastMs < AUTO_MS) return run();
      frame.classList.add("stale");
      button.classList.add("attention");
      say("The input changed: press Predict to update.");
    }
    button.addEventListener("click", run);
    return { run, changed, say };
  }

  // Load each chosen model (downloading on first use) and run it; records times.
  async function runModels(keys, specOf, predict, say, results) {
    let total = 0;
    for (const k of keys) {
      const spec = specOf(k);
      await getTF();
      const W = await getWeights(spec, (f) =>
        say(`Downloading the ${NAMES[k]} weights… ${Math.round(f * 100)} % of ${mb(spec.bytes)}`));
      say(`Computing ${NAMES[k]}…`);
      const t0 = performance.now();
      results[k] = await predict(k, W, spec);
      results[k].ms = performance.now() - t0;
      total += results[k].ms;
    }
    return total;
  }

  function segmented(group, onPick) {
    group.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      group.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      onPick(b.dataset.v);
    }));
  }

  // ------------------------------------------------------------ Dataset 1 page
  function liveD1(cfg) {
    const d1 = cfg.d1, G = d1.grid[2], pal = palette(cfg.colors.phase), hole = holeMask(d1);
    const $ = (id) => document.getElementById("d1l-" + id);
    const fields = ["seed", "gamma", "tau", "sigma", "mean"];
    const slider = $("step"), out = $("step-out"), play = $("play"), frame = $("panels");
    const run = { deeponet: deeponetD1, latent_deeponet: latentD1 };
    let n = 11, u0 = null, preds = {}, predN = 0, panels = {}, timer = null, basis = null;
    colorbar($("cb"), cfg.colors.phase, "‖d‖");
    const bspec = d1.load_basis;
    const getBasis = () => once("basis", () =>
      download(DIR + bspec.file).then((b) => {
        basis = halfToFloat(new Uint16Array(b));
        update();
      }));

    const picker = modelPicker($("models"), Object.keys(run), ["deeponet"],
      (k) => `· ${mb(d1.models[k][n].bytes)}`, () => job.changed());

    const job = runner($("run"), $("status"), frame, async (say) => {
      const keys = picker.chosen();
      if (!keys.length) return null;
      if (!u0) { say("Downloading the load data…"); await getBasis(); }
      const steps = n, input = u0, results = {};
      const ms = await runModels(keys, (k) => d1.models[k][steps],
        (k, W) => run[k](W, d1, hole, input, steps), say, results);
      preds = results;
      if (predN !== steps) { slider.max = steps; slider.value = steps; }
      predN = steps;
      panels = makePanels(frame, keys);
      keys.forEach((k) => { panels[k].caption.innerHTML = `${NAMES[k]}<small>${seconds(results[k].ms)}</small>`; });
      drawStep();
      return ms;
    });

    function params() {
      const p = {};
      fields.forEach((k) => {
        const el = $(k), v = Number(el.value);
        p[k] = Math.min(Number(el.max), Math.max(Number(el.min), Number.isFinite(v) ? v : Number(el.defaultValue)));
      });
      p.seed = Math.round(p.seed);
      return p;
    }

    function update() {
      const p = params(), alpha = boundaryProfile(d1.grf, G, p.seed, p);
      profileChart($("profile"), alpha);
      if (!basis) return;
      u0 = displacement(basis, alpha);
      const mag = new Float64Array(G * G);
      let vmax = 1e-30;
      for (let k = 0; k < G * G; k++) vmax = Math.max(vmax, (mag[k] = Math.hypot(u0[2 * k], u0[2 * k + 1])));
      paint($("input"), G, G, Uint8Array.from(mag, (m, k) => (hole[k] ? 255 : level(m, vmax))), pal);
      job.changed();
    }

    function drawStep() {
      const t = Number(slider.value) - 1, o = t * G * G * 2;
      Object.entries(panels).forEach(([k, p]) => {
        const pred = preds[k], idx = new Uint8Array(G * G);
        for (let a = 0; a < G * G; a++)
          idx[a] = hole[a] ? 255 : level(Math.hypot(pred[o + 2 * a], pred[o + 2 * a + 1]), 1);
        paint(p.canvas, G, G, idx, pal);
      });
      out.textContent = `Load step ${t + 1} / ${predN}`;
    }

    function stop() {
      clearInterval(timer);
      timer = null;
      play.textContent = "▶ Play";
    }

    fields.forEach((k) => $(k).addEventListener("change", update));
    $("dice").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 100000); update(); });
    $("reset").addEventListener("click", () => {
      Object.entries(d1.grf.defaults).forEach(([k, v]) => { $(k).value = v; });
      update();
    });
    segmented($("res"), (v) => { n = Number(v); picker.refresh(); job.changed(); });
    slider.addEventListener("input", () => { stop(); if (predN) drawStep(); });
    play.addEventListener("click", () => {
      if (!predN) return;
      if (timer) return stop();
      if (Number(slider.value) >= predN) slider.value = 1;
      play.textContent = "❚❚ Pause";
      drawStep();
      timer = setInterval(() => {
        if (Number(slider.value) >= predN) return stop();
        slider.value = Number(slider.value) + 1;
        drawStep();
      }, predN > 22 ? 180 : 320);
    });

    update();
    job.say(`Downloading the elasticity solution for the first load step (${mb(2 * product(bspec.shape))})…`);
    getBasis()
      .then(() => job.say("Ready. Choose the surrogates and press Predict."))
      .catch((e) => job.say(`The load data could not be downloaded: ${e.message}`, true));
    getTF().catch(() => {});            // start fetching TensorFlow.js in the background
  }

  // ------------------------------------------------------------ Dataset 2 page
  function liveD2(cfg) {
    const d2 = cfg.d2, [H, X] = d2.shape, pal = palette(cfg.colors.phase);
    const $ = (id) => document.getElementById("d2l-" + id);
    const cfgPal = palette([cfg.colors.plate, cfg.colors.stiffer, cfg.colors.softer, cfg.colors.crack]);
    const tipX = $("tx"), tipY = $("ty"), frame = $("panels");
    const run = { fno: fnoD2, deeponet: deeponetD2, latent_deeponet: latentD2 };
    let upper = false, lower = true, modulus = null, d0 = null;
    colorbar($("cb"), cfg.colors.phase, "d");
    tipX.min = d2.tip_x[0]; tipX.max = d2.tip_x[1]; tipX.value = d2.tip_default[0];
    tipY.min = d2.tip_y[0]; tipY.max = d2.tip_y[1]; tipY.value = d2.tip_default[1];
    const getModulus = () => once("modulus", () =>
      download(DIR + d2.modulus.file).then((b) => new Float32Array(b)));

    const picker = modelPicker($("models"), Object.keys(run), ["deeponet"],
      (k) => `· ${mb(d2.models[k].bytes)}`, () => job.changed());

    const job = runner($("run"), $("status"), frame, async (say) => {
      const keys = picker.chosen();
      if (!keys.length) return null;
      modulus = await getModulus();
      const c = d2.modulus.cases.indexOf(d2.modulus.case_of[`${+upper}${+lower}`]);
      const en = modulus.subarray(c * H * X, (c + 1) * H * X), input = d0, results = {};
      const ms = await runModels(keys, (k) => d2.models[k],
        (k, W, spec) => run[k](W, spec, d2, input, en), say, results);
      const panels = makePanels(frame, keys);
      keys.forEach((k) => {
        paint(panels[k].canvas, H, X, Uint8Array.from(results[k], (v) => level(v, 1)), pal);
        panels[k].caption.innerHTML = `${NAMES[k]}<small>${seconds(results[k].ms)}</small>`;
      });
      return ms;
    });

    function update() {
      const tx = Number(tipX.value), ty = Number(tipY.value);
      $("tx-out").textContent = `${tx} px · ${(tx / 32).toFixed(2)} mm`;
      $("ty-out").textContent = `${ty} px · ${(ty / 32).toFixed(2)} mm`;
      d0 = initialCrack(d2, tx, ty);
      const idx = new Uint8Array(H * X);
      for (let y = 0; y < H; y++) for (let x = 0; x < X; x++) {
        const k = y * X + x;
        idx[k] = d0[k] > 0.3 ? 3
          : inDisc(x, y, d2.upper) ? (upper ? 1 : 2)
            : inDisc(x, y, d2.lower) ? (lower ? 1 : 2) : 0;
      }
      paint($("input"), H, X, idx, cfgPal);
      job.changed();
    }

    segmented($("upper"), (v) => { upper = v === "1"; update(); });
    segmented($("lower"), (v) => { lower = v === "1"; update(); });
    [tipX, tipY].forEach((el) => el.addEventListener("input", update));
    $("input").addEventListener("click", (e) => {    // click the plate to move the tip
      const box = e.currentTarget.getBoundingClientRect();
      tipX.value = Math.round(((e.clientX - box.left) / box.width) * X);
      tipY.value = Math.round((1 - (e.clientY - box.top) / box.height) * H);
      update();
    });

    update();
    job.say("Choose the surrogates and press Predict. Each model's weights are downloaded once.");
    getModulus().catch(() => {});       // a failure is reported when Predict is pressed
    getTF().catch(() => {});
  }

  // ------------------------------------------------------------ mode switch
  const started = {};
  function start(ds) {
    if (started[ds]) return;
    started[ds] = true;
    getConfig()
      .then((cfg) => (ds === "d1" ? liveD1 : liveD2)(cfg))
      .catch((e) => {
        started[ds] = false;
        const s = document.getElementById(ds + "l-status");
        s.textContent = `The live models could not be loaded: ${e.message}`;
        s.classList.add("error");
      });
  }

  document.querySelectorAll("[data-modes]").forEach((group) => {
    const ds = group.dataset.modes;
    segmented(group, (mode) => {
      document.getElementById(ds + "-test").hidden = mode !== "test";
      document.getElementById(ds + "-live").hidden = mode !== "live";
      if (mode === "live") start(ds);
    });
  });

  if (location.search.includes("livetest")) {         // used only to check against Python
    window.__liveTest = { getConfig, getTF, getWeights, download, halfToFloat, holeMask, displacement,
      initialCrack, deeponetD1, latentD1, deeponetD2, latentD2, fnoD2 };
  }
})();
