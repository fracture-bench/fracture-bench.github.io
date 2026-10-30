# Fracture-Bench Playground

The Fracture-Bench website, a static site for GitHub Pages:

- `index.html`: home page with the paper, code, datasets, trained models and poster.
- `playground.html`: an overview of the two datasets, and viewers that show FNO, DeepONet and Latent DeepONet predictions on held-out test cases next to the finite-element (FEM) result, with error maps. A **live** mode lets visitors set up a new case (a new boundary load for Dataset 1; the inclusions and crack tip for Dataset 2) and runs DeepONet on it in their browser.

The test-case predictions were computed once from the trained models and are stored as images in `data/` (28 MB). The live mode uses the DeepONet weights in `live/` (86 MB) with [TensorFlow.js](https://www.tensorflow.org/js), loaded from the jsDelivr CDN. The site runs no Python and needs no server beyond GitHub Pages.

## Publish on GitHub Pages

1. Push this folder to a GitHub repository, for example `Fracture-Bench-Playground`.
2. In the repository, open **Settings → Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/ (root)`, and click **Save**.
3. After a minute or two the site is live at `https://<user-or-organization>.github.io/Fracture-Bench-Playground/`.

`.nojekyll` tells GitHub Pages to serve the files as they are.

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
assets/live.js        live mode: DeepONet in TensorFlow.js, and its controls
data/manifest.json    test cases, colour scales, per-step MSE
data/d1/<sample>/     Dataset 1: input, FEM and each model at 11 / 22 / 44 load steps
data/d2/s<k>/         Dataset 2: input, FEM and each model at the final step
live/live.json        constants, geometry and the byte layout of the weight files
live/deeponet_*.bin   DeepONet weights (float32): Dataset 1 at 11 / 22 / 44 steps, Dataset 2
live/d1_load_basis.bin  u(x,0) for each boundary-load value (float16)
live/d2_modulus.bin   normalized E(x) of the four inclusion cases
```

Each field image is a strip of frames, one per load step, coloured with the app's colormaps.

## Live DeepONet

`assets/live.js` rebuilds the two DeepONet networks of the training code in TensorFlow.js and runs them on the visitor's graphics card (WebGL). A prediction takes well under a second for Dataset 2 and one to a few seconds for Dataset 1. The inputs are built as in the Streamlit app:

- Dataset 1: the boundary load comes from the same Gaussian random field as the FEM data generator. The first load step is linear elastic, so u(x,0) is a weighted sum of the precomputed responses in `d1_load_basis.bin`.
- Dataset 2: the modulus field of the chosen inclusion case, and a pre-crack from the left edge to the chosen tip.

Checked against the Python models: the predictions agree to within 1e-5 (Dataset 2) and 8e-4 (Dataset 1; the difference comes from storing the load basis at half precision).

The weight files are at most 24 MB each, so they can be uploaded through the GitHub website (limit 25 MB per file).

## Regenerate the results

The images come from `tools/export_static_viewer.py` in the Streamlit version of the playground, which contains the model code. With the trained weights available:

```bash
FRACTURE_BENCH_WEIGHTS=<weights folder or URL> python tools/export_static_viewer.py <path to this folder>
```

To add UNet or cDDPM later, add the model to that script and rerun it. The viewer reads its list of models from `data/manifest.json`.

The files in `live/` come from `tools/export_live_deeponet.py` in the same folder:

```bash
FRACTURE_BENCH_WEIGHTS=<weights folder or URL> python tools/export_live_deeponet.py <path to this folder>
```

The test cases come from the Fracture-Bench datasets, released under CC BY 4.0.
