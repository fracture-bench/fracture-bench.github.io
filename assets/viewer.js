// Playground: tabs, plus viewers for the precomputed predictions in data/.
// Field images are horizontal strips of frames; a panel shows one frame by
// shifting the strip with background-position.

// ------------------------------------------------------------ tabs
const tabs = [...document.querySelectorAll(".tab")];

function selectTab(name, focus) {
  const tab = tabs.find((t) => t.dataset.tab === name) || tabs[0];
  tabs.forEach((t) => {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  });
  if (focus) tab.focus();
  history.replaceState(null, "", "#" + tab.dataset.tab);
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t.dataset.tab));
  t.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (step) selectTab(tabs[(i + step + tabs.length) % tabs.length].dataset.tab, true);
  });
});
document.querySelectorAll("[data-goto]").forEach((b) =>
  b.addEventListener("click", () => {
    selectTab(b.dataset.goto);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }));
selectTab(location.hash.slice(1) || "overview");
window.addEventListener("hashchange", () => selectTab(location.hash.slice(1)));

// ------------------------------------------------------------ helpers
const FEM_LABEL = "FEM (ground truth)";

function fmt(x) {
  return x.toExponential(1).replace("e-", "e−");
}

function showFrame(el, url, count, index) {
  el.style.backgroundImage = `url("${url}")`;
  el.style.backgroundSize = `${count * 100}% 100%`;
  el.style.backgroundPosition = count > 1 ? `${(index / (count - 1)) * 100}% 0` : "0 0";
}

function panel(label) {
  const fig = document.createElement("figure");
  fig.className = "panel";
  fig.innerHTML = '<div class="frame" role="img"></div><figcaption></figcaption>';
  fig.frame = fig.firstChild;
  fig.caption = fig.lastChild;
  fig.frame.setAttribute("aria-label", label);
  return fig;
}

function colorbar(el, stops, max, label) {
  el.innerHTML = "";
  const lo = document.createElement("span");
  lo.textContent = "0";
  const grad = document.createElement("div");
  grad.className = "grad";
  grad.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
  const hi = document.createElement("span");
  hi.textContent = max;
  const name = document.createElement("em");
  name.textContent = label;
  el.append(lo, grad, hi, name);
}

function modelChecks(container, models, onChange) {
  container.innerHTML = "";
  models.forEach((m) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.value = m.key;
    box.addEventListener("change", onChange);
    label.append(box, document.createTextNode(m.name));
    container.append(label);
  });
  return () => [...container.querySelectorAll("input:checked")].map((b) => b.value);
}

function preload(urls) {
  urls.forEach((u) => { const img = new Image(); img.src = u; });
}

// ------------------------------------------------------------ Dataset 1
function datasetOne(M) {
  const S = M.d1.samples;
  const names = Object.fromEntries(M.models.map((m) => [m.key, m.name]));
  const sel = document.getElementById("d1-sample");
  const input = document.getElementById("d1-input");
  const res = document.getElementById("d1-res");
  const slider = document.getElementById("d1-step");
  const out = document.getElementById("d1-step-out");
  const play = document.getElementById("d1-play");
  const pred = document.getElementById("d1-pred");
  const err = document.getElementById("d1-err");
  const cbErr = document.getElementById("d1-cb-err");
  colorbar(document.getElementById("d1-cb-phase"), M.colormaps.phase, "1", "‖d‖");

  S.forEach((s, i) => sel.add(new Option(`Sample #${s.id} (test split)`, i)));
  const chosen = modelChecks(document.getElementById("d1-models"), M.models, build);
  let n = 11, timer = null, panels = {};

  function dir() { return `data/d1/${S[sel.value].id}`; }

  function build() {
    stop();
    const keys = chosen();
    pred.innerHTML = "";
    err.innerHTML = "";
    panels = {};
    if (!keys.length) {
      pred.innerHTML = '<p class="empty">Select at least one surrogate.</p>';
      draw();
      return;
    }
    keys.forEach((k) => {
      panels[k] = { pred: panel(`${names[k]} prediction`), err: panel(`${names[k]} error`) };
      pred.append(panels[k].pred);
      err.append(panels[k].err);
    });
    panels.fem = panel(FEM_LABEL);
    pred.append(panels.fem);
    preload([`${dir()}/truth.png`, ...keys.flatMap((k) => [`${dir()}/${k}_${n}.png`, `${dir()}/${k}_${n}_err.png`])]);
    draw();
  }

  function draw() {
    const s = S[sel.value];
    const t = Number(slider.value);
    out.textContent = `Load step ${t} / ${n}`;
    const vmax = s.err_vmax[n][t - 1];
    colorbar(cbErr, M.colormaps.error, fmt(vmax), "|pred − FEM|");
    Object.entries(panels).forEach(([k, p]) => {
      if (k === "fem") {
        showFrame(p.frame, `${dir()}/truth.png`, 44, (t - 1) * (44 / n));
        p.caption.textContent = FEM_LABEL;
        return;
      }
      showFrame(p.pred.frame, `${dir()}/${k}_${n}.png`, n, t - 1);
      showFrame(p.err.frame, `${dir()}/${k}_${n}_err.png`, n, t - 1);
      p.pred.caption.textContent = names[k];
      p.err.caption.innerHTML = `${names[k]}<small>MSE ${fmt(s.mse[n][k][t - 1])}</small>`;
    });
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    play.textContent = "▶ Play";
  }

  sel.addEventListener("change", () => { input.src = `${dir()}/input.png`; build(); });
  res.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      n = Number(b.dataset.n);
      res.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      slider.max = n;
      slider.value = n;
      build();
    }));
  slider.addEventListener("input", () => { stop(); draw(); });
  play.addEventListener("click", () => {
    if (timer) return stop();
    if (Number(slider.value) >= n) slider.value = 1;
    play.textContent = "❚❚ Pause";
    draw();
    timer = setInterval(() => {
      if (Number(slider.value) >= n) return stop();
      slider.value = Number(slider.value) + 1;
      draw();
    }, n > 22 ? 180 : 320);
  });

  input.src = `${dir()}/input.png`;
  build();
}

// ------------------------------------------------------------ Dataset 2
function datasetTwo(M) {
  const S = M.d2.samples;
  const names = Object.fromEntries(M.models.map((m) => [m.key, m.name]));
  const sel = document.getElementById("d2-sample");
  const input = document.getElementById("d2-input");
  const pred = document.getElementById("d2-pred");
  const err = document.getElementById("d2-err");
  colorbar(document.getElementById("d2-cb-phase"), M.colormaps.phase, "1", "d");

  const word = (stiff) => (stiff ? "stiffer" : "softer");
  S.forEach((s, i) => sel.add(new Option(
    `Case ${i + 1}: upper ${word(s.upper_stiffer)}, lower ${word(s.lower_stiffer)}`, i)));
  const chosen = modelChecks(document.getElementById("d2-models"), M.models, build);

  function build() {
    const s = S[sel.value];
    const d = `data/d2/${s.dir}`;
    input.src = `${d}/input.png`;
    pred.innerHTML = "";
    err.innerHTML = "";
    const keys = chosen();
    colorbar(document.getElementById("d2-cb-err"), M.colormaps.error, fmt(s.err_vmax), "|pred − FEM|");
    if (!keys.length) pred.innerHTML = '<p class="empty">Select at least one surrogate.</p>';
    keys.forEach((k) => {
      const p = panel(`${names[k]} prediction`);
      const e = panel(`${names[k]} error`);
      showFrame(p.frame, `${d}/${k}.png`, 1, 0);
      showFrame(e.frame, `${d}/${k}_err.png`, 1, 0);
      p.caption.textContent = names[k];
      e.caption.innerHTML = `${names[k]}<small>MSE ${fmt(s.mse[k])}</small>`;
      pred.append(p);
      err.append(e);
    });
    if (keys.length) {
      const f = panel(FEM_LABEL);
      showFrame(f.frame, `${d}/truth.png`, 1, 0);
      f.caption.textContent = FEM_LABEL;
      pred.append(f);
    }
  }
  sel.addEventListener("change", build);
  build();
}

// ------------------------------------------------------------ start
fetch("data/manifest.json")
  .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then((M) => { datasetOne(M); datasetTwo(M); })
  .catch(() => {
    document.querySelectorAll("[data-viewer] .output").forEach((o) => {
      o.innerHTML = '<p class="empty">The precomputed results could not be loaded. If you opened this file directly, serve the folder over HTTP instead (see README).</p>';
    });
  });
