// Public links. While a value is empty, its button reads "Coming soon".
const LINKS = {
  paper: "",   // preprint or journal page
  code: "https://github.com/Centrum-IntelliPhysics/fracture_bench",  // GitHub repository with the training code
  models: "",  // public share link of the folder with the trained weights
  poster: "",  // poster PDF, e.g. "assets/Fracture-Bench_poster.pdf"
};

// ------------------------------------------------------------ links
document.querySelectorAll("[data-link]").forEach((el) => {
  const url = LINKS[el.dataset.link];
  if (url) {
    el.href = url;
    el.target = "_blank";
    el.rel = "noopener";
    return;
  }
  if (el.dataset.optional !== undefined) {   // e.g. the training-code button
    el.hidden = true;
    return;
  }
  el.removeAttribute("href");
  el.classList.add("is-soon");
  el.setAttribute("aria-disabled", "true");
  el.textContent = "Coming soon";
});
