# Project Deliverable Templates

LaTeX templates for the written project deliverables, based on the official NeurIPS style. Grading, deadlines and submission rules are in the [course modalities](../modalities.md).

| Folder | Deliverable | Page limit (excluding references) |
|---|---|---|
| [`proposal/`](proposal/) | Project proposal | 4 pages |
| [`report/`](report/) | Project report | 9 pages |

Each template contains all required sections. A grey box under each heading describes what the section must contain.

## How to use a template

1. **Overleaf (recommended):** download the folder as a ZIP file, then in Overleaf choose *New Project → Upload Project*.

   **Locally:** compile `main.tex`, e.g. with `latexmk -pdf main.tex`.
2. Replace the title, the authors and every "Your text." with your content. Add the sources you cite to `references.bib`.
3. Before submitting, hide the grey boxes: change `\guidancetrue` to `\guidancefalse` at the top of `main.tex`.
4. Check the page limit, then submit the PDF in Moodle.

Keep the `preprint` option in `\usepackage[preprint]{neurips_2025}`, so that your names appear on the paper. Do not change margins, font sizes or spacing.
