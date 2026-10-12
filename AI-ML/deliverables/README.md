# Project Guide

This guide describes how to find a project topic and how to prepare each project deliverable of the **AI and Machine Learning** course. The rules on teams, grading, deadlines and W&B tracking are in the [course modalities](../modalities.md).

| Deliverable | Template | Limit |
|---|---|---|
| Project proposal | **[Open in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Fproposal%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Fproposal%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Fproposal%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex)** · [files](proposal/) | 4 pages, excluding references |
| Project report | **[Open in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Freport%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Freport%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Freport%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex)** · [files](report/) | 9 pages, excluding references |
| Oral defensio | any tool (Section 6) | 10-minute team presentation + 20-minute oral examination |

---

## 1. Finding a Topic

Your team needs one dataset and one learning task $(T, P, E)$ that all three of you work on, each with a different model family. Good starting points are tabular datasets from real application domains, e.g. credit card fraud detection, customer churn or medical diagnosis. These websites list datasets, tasks and published results:
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) - classic tabular datasets with documentation
- [OpenML](https://www.openml.org/) - datasets with benchmark results of many algorithms
- [Kaggle](https://www.kaggle.com/) - see the *Datasets* and *Competitions* tabs
- [Papers with Code](https://paperswithcode.co) - tasks, benchmarks and leaderboards with linked papers and code
- [Google Scholar](https://scholar.google.com/) or [Semantic Scholar](https://www.semanticscholar.org/) - publications for your previous-work section

You may use any material you find on the Web. The only requirement is that **you understand and can explain the approach, the data and the results**.

> 📝 **Example:** A team picks customer churn prediction on a public telecom dataset. The task is binary classification: will a customer cancel the contract? Because churners are a minority, the team measures success with the $F_1$-score and ROC-AUC instead of accuracy. Student A trains a regularized logistic regression, Student B a gradient-boosted tree ensemble, and Student C a support vector machine with an RBF kernel; the report compares the three with each other and with published results.

Please get in touch with the lecturer if you have any questions!

---

## 2. Paper Format & Templates

Write the proposal and the report like papers for a machine learning conference, using the LaTeX templates in [`proposal/`](proposal/) and [`report/`](report/). They are based on the official NeurIPS style and contain all required sections. The LaTeX comments (`% ...`) under each heading describe what the section must contain; as comments, they never appear in your PDF.

**How to use a template:**
1. **Open the template in Overleaf:** click [Open the proposal template in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Fproposal%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Fproposal%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Fproposal%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex) or [Open the report template in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Freport%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Freport%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FAI-ML%2Fdeliverables%2Freport%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex). Log in to [Overleaf](https://www.overleaf.com/), or create a free account. Overleaf creates your own copy of the template as a new project, ready to compile. You don't need to install LaTeX.
2. **Share it with your team by link:** click *Share* in Overleaf, turn on link sharing, and send the **edit link** to your team members, so that you can write together. We recommend link sharing instead of invitations by email.
3. **Write:** replace the title, the authors and every "Your text." with your content, following the comments under each heading. Add the sources you cite to `references.bib`.
4. **Submit:** check the page limit, then download the PDF from Overleaf and submit it in Moodle.

If you prefer to work locally, download the files from the template folder and compile `main.tex`, e.g. with `latexmk -pdf main.tex`.

**Rules:**
- Keep the `preprint` option in `\usepackage[preprint]{neurips_2025}`, so that your names appear on the paper.
- **Page limits**, **excluding references**:
  - **proposal:** at most **4 pages**;
  - **report:** at most **9 pages**.

  Everything except the references counts towards the limit: figures, tables, appendices and the AI disclosure statement. Do not change margins, font sizes or spacing to fit more content.
- **Attribution:** state in every section who is expected to do, or has done, what, e.g. "The logistic regression baseline was implemented and evaluated by *(name)* …".

---

## 3. Project Proposal

Write your proposal with the [proposal template](proposal/) and submit it as a **PDF** of at most 4 pages. It contains these five sections, **strictly in this order**, and the AI disclosure statement in the appendix:
1. *Motivation and Domain Problem*
2. *Learning Task* $(T, P, E)$
3. *Previous Work and Literature Baselines*, with at least 3 related publications, benchmarks or authoritative studies
4. *Implementation Plan and Individual Roles*, including the distribution of the three model families across the team
5. *Risk Management*

The template describes the questions each section must answer.

---

## 4. Project Report

Write your report with the [report template](report/). It is a scientific paper of at most **9 pages** with these sections:
- Abstract and Motivation
- Data Exploration and Preprocessing, with leak-free `sklearn.pipeline.Pipeline` objects
- Theoretical Background and Model Selection, with the mathematical formulation of each model family
- Experimental Setup and Hyperparameter Optimization
- Evaluation and Diagnostics, with **one consolidated table** comparing the models of all team members with the literature baselines
- Error Analysis and Discussion
- Conclusions and Future Directions
- References
- Appendix, including the AI disclosure statement

The template describes what each section must contain.

### Submission

Submit in Moodle:
1. the report as a **PDF**;
2. **your code**, in one of the two forms described in Section 5: either an archive, or a link to your repository or Colab notebook in the report.

The report must also include the **public link to your team's W&B workspace**; the W&B rules are in the [course modalities](../modalities.md).

---

## 5. Code & Reproducibility

Your code must allow us to reproduce the main results of your report. There are **two possibilities** to provide it:

1. **A code archive or repository**, e.g. a ZIP file or a link to a GitHub repository, containing:
   - **a `README.md`** that explains how to set up the environment, obtain the data, train your models, and reproduce the tables and figures of your report;
   - **an environment specification** with pinned package versions and the Python version, e.g. `requirements.txt`, or `pyproject.toml` with a lock file;
   - **data loaders and a clear entry point** for each main result, e.g. `train.py` and `evaluate.py`, or a notebook that runs from top to bottom.
2. **A Google Colab notebook** that runs from top to bottom.
   - It installs the packages with pinned versions, downloads the data and the saved models, and reproduces your main results.
   - Text cells explain each step.
   - Share it with *Anyone with the link can view* and put the link in the report.

In both cases, provide:
- **The data, or how to get it:** include the data if it is small; otherwise, provide a download link or script. Always include the preprocessing code.
- **Fixed random seeds**, e.g. `random_state` in scikit-learn and the NumPy seed, so that runs can be repeated.
- **The saved models** of your final pipelines, e.g. stored with `joblib`, or a link to them.

> ⚠️ **Important:** When you work in Google Colab, everything stored in the runtime (`/content`) is **deleted when the session ends**. Connect Google Drive (`from google.colab import drive; drive.mount("/content/drive")`) and save your data, notebooks, models and results there.

---

## 6. Oral Defensio

The defensio is an **in-person examination**. The room is announced in Moodle.

- **10-minute team presentation:** 5–6 content slides summarizing the task, the data, the methodology, the comparative results and the conclusions. You can prepare the slides with any tool, e.g. PowerPoint, LaTeX Beamer or Jupyter RISE slides; upload them to Moodle as a **PDF**.
- **20-minute oral examination:** targeted questions to each student individually.

### What the examiners focus on

- **Mathematical principles:** loss functions, gradient updates, decision boundary geometry, regularization penalties.
- **Justification of your choices:** why you selected specific algorithms, hyperparameters and feature encodings.
- **Pipeline integrity:** preventing data leakage, understanding the variance across cross-validation folds, and diagnosing overfitting.
- **Interpretation of results:** the meaning of your metrics, trade-offs between precision and recall, and the limitations of your models.
