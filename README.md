# Fracture-Bench Playground

The Fracture-Bench website, a static site for GitHub Pages:

- `index.html`: home page with the paper, code, datasets, trained models and poster.
- `playground.html`: an overview of the two datasets, and viewers that show FNO, DeepONet and Latent DeepONet predictions on held-out test cases next to the finite-element (FEM) result, with error maps. A **live** mode lets visitors set up a new case (a new boundary load for Dataset 1; the inclusions and crack tip for Dataset 2) and runs the surrogates on it in their browser: DeepONet and Latent DeepONet for Dataset 1, and all three for Dataset 2.

The test-case predictions were computed once from the trained models and are stored as images in `data/` (28 MB). The live mode uses the model weights in `live/` (382 MB; each visitor downloads only the models they pick) with [TensorFlow.js](https://www.tensorflow.org/js), loaded from the jsDelivr CDN. The site runs no Python and needs no server beyond GitHub Pages.

## Publish on GitHub Pages

1. Push this folder to a GitHub repository, for example `Fracture-Bench-Playground`.
2. In the repository, open **Settings → Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/ (root)`, and click **Save**.
3. After a minute or two the site is live at `https://<user-or-organization>.github.io/Fracture-Bench-Playground/`.

`.nojekyll` tells GitHub Pages to serve the files as they are.

The pages load `assets/*.js` and `assets/style.css` with a version tag, e.g. `assets/site.js?v=20261031`. After changing any of those files, change the tag in `index.html` and `playground.html` (to the date, say) so browsers fetch the new copies instead of old saved ones.

## Add the links

The links for the paper, training code and trained models are at the top of `assets/site.js`:

```js
const LINKS = {
  paper: "",   // preprint or journal page
  code: "",    // GitHub repository with the training code
  models: "",  // Google Drive folder with the trained weights
  poster: "",  // poster PDF, e.g. "assets/Fracture-Bench_poster.pdf"
};
```

While a link is empty, its card is faded with a "Coming soon" badge. To publish the poster, copy the PDF into `assets/` and set `poster` to its path (the training-code button is hidden).

## Preview locally

`fetch()` does not work on pages opened straight from disk, so serve the folder:

```bash
python -m http.server 8600
```

Then open http://localhost:8600.

## Layout

```
index.html, playground.html
assets/style.css      design (white pages, dark-blue text, blue headings)
assets/site.js        the paper, code and model links
assets/viewer.js      tabs and the two viewers
assets/live.js        live mode: the surrogates in TensorFlow.js, and their controls
data/manifest.json    test cases, colour scales, per-step MSE
data/d1/<sample>/     Dataset 1: input, FEM and each model at 11 / 22 / 44 load steps
data/d2/s<k>/         Dataset 2: input, FEM and each model at the final step
live/live.json        constants, geometry and the byte layout of the weight files
live/<model>_d1_<n>.bin  Dataset 1 weights, n = 11 / 22 / 44 load steps (DeepONet, Latent DeepONet)
live/<model>_d2*.bin  Dataset 2 weights (FNO, DeepONet, Latent DeepONet); files over 45 MB come in parts .1, .2
live/d1_load_basis.bin  u(x,0) for each boundary-load value (float16)
live/d2_modulus.bin   normalized E(x) of the four inclusion cases
```

Each field image is a strip of frames, one per load step, coloured with the app's colormaps.

## Live models

`assets/live.js` rebuilds the trained networks in TensorFlow.js and runs them on the visitor's graphics card (WebGL): DeepONet and Latent DeepONet for Dataset 1, and FNO, DeepONet and Latent DeepONet for Dataset 2. The Dataset 1 FNO (1.2 GB per model) is too large for a browser and is shown for the held-out test cases only. The inputs are built as in the Streamlit app:

- Dataset 1: the boundary load comes from the same Gaussian random field as the FEM data generator. The first load step is linear elastic, so u(x,0) is a weighted sum of the precomputed responses in `d1_load_basis.bin`.
- Dataset 2: the modulus field of the chosen inclusion case, and a pre-crack from the left edge to the chosen tip.

To keep downloads and computation small, a few layers are rearranged (exactly, up to rounding): DeepONet's last trunk layer is folded into the branch output; Latent DeepONet's Dataset 1 operator is folded into one matrix when exporting; and the FNO's Fourier transforms, which keep only 32 x 48 modes, become small matrix products. The largest matrices of Latent DeepONet and FNO are stored at half precision.

Checked against the Python models, the predictions agree to within 8e-4 for every model (one colour level is 4e-3). A prediction takes 0.1–0.3 s for Dataset 2 and about a second for Dataset 1 on a laptop.

Each weight file is under 50 MB, within GitHub's file-size limits.

## Regenerate the results

The images come from `tools/export_static_viewer.py` in the Streamlit version of the playground, which contains the model code. With the trained weights available:

```bash
FRACTURE_BENCH_WEIGHTS=<weights folder or URL> python tools/export_static_viewer.py <path to this folder>
```

To add UNet or cDDPM later, add the model to that script and rerun it. The viewer reads its list of models from `data/manifest.json`.

The files in `live/` come from `tools/export_live_models.py` in the same folder:

```bash
FRACTURE_BENCH_WEIGHTS=<weights folder or URL> python tools/export_live_models.py <path to this folder>
```

The test cases come from the Fracture-Bench datasets, released under CC BY 4.0.
