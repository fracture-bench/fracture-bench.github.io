# Fracture-Bench Playground

The Fracture-Bench website, a static site for GitHub Pages:

- `index.html`: home page with the paper, code, datasets, trained models and poster.
- `playground.html`: an overview of the two datasets, and viewers that show FNO, DeepONet and Latent DeepONet predictions on held-out test cases next to the finite-element (FEM) result, with error maps.

The predictions were computed once from the trained models and are stored as images in `data/` (28 MB). The site runs no Python and needs no server beyond GitHub Pages.

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
data/manifest.json    test cases, colour scales, per-step MSE
data/d1/<sample>/     Dataset 1: input, FEM and each model at 11 / 22 / 44 load steps
data/d2/s<k>/         Dataset 2: input, FEM and each model at the final step
```

Each field image is a strip of frames, one per load step, coloured with the app's colormaps.

## Regenerate the results

The images come from `tools/export_static_viewer.py` in the Streamlit version of the playground, which contains the model code. With the trained weights available:

```bash
FRACTURE_BENCH_WEIGHTS=<weights folder or URL> python tools/export_static_viewer.py <path to this folder>
```

To add UNet or cDDPM later, add the model to that script and rerun it. The viewer reads its list of models from `data/manifest.json`.

The test cases come from the Fracture-Bench datasets, released under CC BY 4.0.
