# Paper: Adaptive latent reasoning

This branch holds the written report of the project (ACL format, in Spanish):
*"Razonamiento latente adaptativo: halting probabilistico tipo PonderNet sobre
cadenas de pensamiento implicitas"*.

```
paper/
├── main.tex         # The report (ACL format, final mode) — named main.tex so
                     #   Overleaf picks it up as the main document automatically
├── paper.md         # Internal editorial notes (not part of the report)
├── custom.bib       # Bibliography
├── acl.sty          # ACL style, vendored so the branch compiles standalone
├── acl_natbib.bst   # ACL bibliography style
├── figures/         # Figures (step-count distribution)
└── results_test/    # Held-out test eval artifacts backing the numbers
                     #   (M0/M1/M2 adaptive operating points + fixed-K baseline)
```

The report covers the three project directions: the upfront k* classifier (Option A),
the PonderNet halting method (Option C, the core contribution), and the adaptive
vectors-per-step c-axis (Option B). All Option C metrics are reported on the held-out
test split (n=1319); the three adaptive operating points M0/M1/M2 are the same model at
three gamma / prior-shape settings (see `results_test/frontier_test.md`).

For the current integrated state of the project, see the `main` branch; for the final
state of each approach, see `option-a-k-classifier`, `option-b-adaptive-vectors`, and
`option-c-pondernet`.

## Building

The ACL style files are vendored in `paper/`, so the branch compiles with a standard
TeX installation. On Overleaf: upload the contents of `paper/` (or the whole folder) to
a project; since the report is named `main.tex`, Overleaf sets it as the main document
automatically, no manual configuration needed. The figure uses `pgfplots`/`tikz`,
already included in a full TeX distribution.

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
