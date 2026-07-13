# report-src

LaTeX sources for the **written report and poster** of the
`adaptive-latent-reasoning` project (UdeSA NLP final project).

The compiled PDFs and the rendered poster live on the `main` branch under
[`report/`](https://github.com/famatodlr/adaptive-latent-reasoning/tree/main/report).
This branch holds the editable sources; the code lives on `main` and the
`option-*` branches.

## Layout

| Folder | What |
|--------|------|
| `paper_en/`  | Report, **English** (ACL format) - build `main.tex` |
| `paper/`     | Report, original **Spanish** |
| `poster_en/` | A0 poster, **English** - build `poster.tex` |
| `poster/`    | A0 poster, original **Spanish** |

Each English / Spanish pair shares the same figures and results. The figure
generation scripts live under `paper*/figures/scripts/` (a shared `figstyle.py`
plus one script per figure); the raw eval artifacts they read from are under
`paper*/results_test/`.

## Build

```bash
# report (English)
cd paper_en && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# poster (English)
cd poster_en && pdflatex poster.tex && pdflatex poster.tex
```
