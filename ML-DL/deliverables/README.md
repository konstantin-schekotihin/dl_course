# Project Guide

This guide describes how to find a project topic and how to prepare each project deliverable. The rules on group work, grading and deadlines are in the [course modalities](../modalities.md).

| Deliverable | Template | Page limit (excluding references) |
|---|---|---|
| Project proposal | **[Open in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Fproposal%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Fproposal%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Fproposal%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex)** · [files](proposal/) | 4 pages |
| Project report | **[Open in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Freport%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Freport%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Freport%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex)** · [files](report/) | 9 pages |
| Project presentation | any tool (Section 7) | 15 minutes |

---

## 1. Finding a Topic

Start with a domain that interests you, for example computer vision. Then identify a task in that domain, such as image classification, object detection, image generation or image modification. These websites list tasks, datasets and approaches to solve them:
- [PapersWithCode](https://paperswithcode.co) (**Recommended**) - tasks, benchmarks and leaderboards with linked papers and code
- [Kaggle](https://www.kaggle.com/) - see the *Competitions* and *Datasets* tabs
- [Hugging Face](https://huggingface.co/) - see the *Models* and *Datasets* tabs
- [Google Scholar](https://scholar.google.com/) or [Semantic Scholar](https://www.semanticscholar.org/)

On paper-listing sites, solutions are described in scientific publications. On Kaggle, they are described in notebooks written by professionals, students and enthusiasts. You may use any material you find on the Web. The only requirement is that **you understand and can explain the approach, the data and the results**.

> 📝 **Example:** A group interested in medical imaging picks pneumonia detection from chest X-rays, using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset on Kaggle. Given an X-ray image, the task is to classify it as *normal* or *pneumonia*. Kaggle lists many solutions to this problem as notebooks. A typical solution:
> - looks at example images and the class balance, and decides on preprocessing and data augmentation;
> - trains a convolutional network from scratch and fine-tunes a network pre-trained on ImageNet;
> - compares the models using precision, recall and ROC-AUC, since a missed pneumonia costs more than a false alarm.
>
> In a group, the members could split this into a plain CNN, a ResNet and a study of the training procedure; see the worked example in Section 4.

Excellent deliverables of students from previous years are available in the **Examples** folder in Moodle. Feel free to use them as a guide.

Please get in touch with the lecturer if you have any questions!

---

## 2. Paper Format & Templates

Write the proposal and the report like papers for a machine learning conference, using the LaTeX templates in [`proposal/`](proposal/) and [`report/`](report/). They are based on the official NeurIPS style and contain all required sections. The LaTeX comments (`% ...`) under each heading describe what the section must contain; as comments, they never appear in your PDF.

**How to use a template:**
1. **Open the template in Overleaf:** click [Open the proposal template in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Fproposal%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Fproposal%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Fproposal%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex) or [Open the report template in Overleaf](https://www.overleaf.com/docs?snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Freport%2Fmain.tex&snip_name%5B%5D=main.tex&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Freport%2Freferences.bib&snip_name%5B%5D=references.bib&snip_uri%5B%5D=https%3A%2F%2Fraw.githubusercontent.com%2Fkonstantin-schekotihin%2Fdl_course%2Fmaster%2FML-DL%2Fdeliverables%2Freport%2Fneurips_2025.sty&snip_name%5B%5D=neurips_2025.sty&main_document=main.tex&engine=pdflatex). Log in to [Overleaf](https://www.overleaf.com/), or create a free account. Overleaf creates your own copy of the template as a new project, ready to compile. You don't need to install LaTeX.
2. **Share it with your group by link:** click *Share* in Overleaf, turn on link sharing, and send the **edit link** to your group members, so that you can write together. We recommend link sharing instead of invitations by email.
3. **Write:** replace the title, the authors and every "Your text." with your content, following the comments under each heading. Add the sources you cite to `references.bib`.
4. **Submit:** check the page limit, then download the PDF from Overleaf and submit it in Moodle.

If you prefer to work locally, download the files from the template folder and compile `main.tex`, e.g. with `latexmk -pdf main.tex`.

**Rules:**
- Keep the `preprint` option in `\usepackage[preprint]{neurips_2025}`, so that your names appear on the paper.
- **Page limits**, **excluding references**:
  - **proposal:** at most **4 pages**, like a short (workshop) paper;
  - **report:** at most **9 pages**, like a full NeurIPS paper.

  Everything except the references counts towards the limit: figures, tables, appendices and the AI disclosure statement. Do not change margins, font sizes or spacing to fit more content.

---

## 3. Project Proposal

Write your proposal with the [proposal template](proposal/) and submit it as a **PDF** of at most 4 pages. The template contains the five required sections: *Motivation*, *Learning Task*, *Previous Work*, *Plan* and *Risk Management*. It also describes the questions each section must answer, and you must answer all of them. The *Plan* section states each member's approach, their **ablation-study plan** and the **plan of the comparative analysis** (Section 4).

If you have doubts about your proposal, contact the lecturer well before the deadline!

---

## 4. Project Report

Write your report with the [report template](report/). It is a research paper of at most **9 pages**, with the structure of a typical machine learning paper:
- Abstract and Introduction
- Related Work, Data, Method and Experimental Setup
- Results, including the **comparative analysis** of all approaches
- Analysis, i.e. the **ablation studies** of all members and error analysis

> ⚠️ **Important:** In the Results section, you **must compare every approach against a baseline**: at least one simple baseline, and one from the literature, blogs, Kaggle, etc., if available. Evaluate the approaches and baselines on the same test data with the same measures. A report without a baseline comparison is incomplete.
- Limitations and Conclusions
- Author Contributions and References
- Appendix, including the AI disclosure statement

The template describes what each section must contain.

### Ablation studies and comparative analysis

Every report contains two kinds of experimental analysis. Plan both in the *Plan* section of your proposal, and report them in the *Results* and *Analysis* sections of your report.

- **Ablation study (each member):** which components of your own approach actually contribute to the result?
  - Remove or replace **one component at a time**, keep everything else fixed, and compare with the full approach.
  - Report it as a table: the full approach first, then one row per removed or replaced component, with the change in the main measure.
  - Connect each finding to the lecture. For example, a deep network without skip connections trains worse because of vanishing gradients.
- **Comparative analysis (whole group):** as the result of the project, compare the approaches of all members with each other and with the baselines:
  - the same data split, test data, evaluation measures and random seeds;
  - a comparable training budget, e.g. the same number of epochs;
  - the number of parameters and the training time of each approach;
  - an explanation of the differences: which approach works best, why, and at what cost? Which examples do all approaches get wrong?

**Worked example:** pneumonia detection from chest X-rays (Section 1), for a group of three.

| Member | Approach | Ablation study |
|---|---|---|
| A | **Plain CNN** trained from scratch: blocks of convolution, batch normalization, ReLU and pooling | without batch normalization · without data augmentation · 2 vs 4 convolutional blocks |
| B | **ResNet-18**, pre-trained on ImageNet and fine-tuned | without skip connections (same depth) · random initialization instead of pre-training · frozen backbone vs full fine-tuning |
| C | **Training-procedure study** on a fixed network, e.g. the plain CNN of Student A | SGD vs SGD with momentum vs Adam · constant vs cosine learning-rate schedule · without weight decay · without early stopping |

The comparative analysis of this group contains:
- one table with test precision, recall and ROC-AUC (mean ± standard deviation over 3 random seeds), parameter count and training time for A's CNN, B's ResNet, C's best training recipe and the baselines;
- the learning curves of all three approaches;
- the X-rays that all three approaches misclassify;
- a justified recommendation, e.g.: is the ResNet's improvement worth its size and training time, or does a well-trained plain CNN suffice?

### Submission

Submit in Moodle:
1. the report as a **PDF**;
2. **your code**, in one of the two forms described in Section 5: either an archive, or a link to your repository or Colab notebook in the report. If an archive is too large, publish it elsewhere (e.g. on GitHub) and put the link in the report.

Paste the link to your public **Weights & Biases (W&B) project** (Section 6) into the submission comments, because Moodle does not open links from the report review page.

> ⚠️ **Important:** Reports that do not meet these requirements will not be considered!

---

## 5. Code & Reproducibility

Your code must allow us to reproduce the main results of your report. There are **two possibilities** to provide it:

1. **A code archive or repository**, e.g. a ZIP file or a link to a GitHub repository, containing:
   - **a `README.md`** that explains how to set up the environment, obtain the data, train your models, and reproduce the tables and figures of your report;
   - **an environment specification** with pinned package versions and the Python version, e.g. `requirements.txt`, or `pyproject.toml` with a lock file;
   - **a clear entry point** for each main result, e.g. `train.py` and `evaluate.py`, or a notebook that runs from top to bottom.
2. **A Google Colab notebook** that runs from top to bottom.
   - It installs the packages with pinned versions, downloads the data and trained weights, and reproduces your main results.
   - Text cells explain each step.
   - Share it with *Anyone with the link can view* and put the link in the report.

In both cases, provide:
- **The data, or how to get it:**
  - Include the data if it is small.
  - Otherwise, provide a download link or script.
  - Always include the preprocessing code. Processed data on its own, without the code that produced it, is not enough.
- **Fixed random seeds** for Python, NumPy and PyTorch, so that runs can be repeated.
- **The trained weights** of your final models, or a link to them, so that the evaluation can be rerun without retraining.

Your evaluation should run on Google Colab or a standard laptop. If training needs more resources, state the hardware you used and the training time in the README or the notebook.

### Training on Google Colab

[Google Colab](https://colab.research.google.com/) gives you **free GPU access**, which speeds up training deep networks considerably. To use it, select *Runtime → Change runtime type → T4 GPU*. Free GPU time is limited, and Colab disconnects sessions after a period of inactivity or a maximum runtime, so plan longer training runs accordingly.

> ⚠️ **Important:** Everything stored in the Colab runtime (`/content`) is **deleted when the session ends**, including downloaded data, trained models and results. **Always store your files and checkpoints on Google Drive** to avoid losing your work.

Mount your Drive at the start of the notebook and save all files there:

```python
from google.colab import drive
drive.mount("/content/drive")

CHECKPOINT_DIR = "/content/drive/MyDrive/dl_project/checkpoints"
```

- **Save checkpoints regularly during training**, e.g. after every epoch: model weights, optimizer state and the current epoch (`torch.save({...}, f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")`). Then you can resume training after a disconnect instead of starting again.
- **Keep your data, notebooks and results on Drive as well**, so that they survive the end of the session.

---

## 6. Experiment Tracking with Weights & Biases

Track **all experiments** of your project with [Weights & Biases (W&B)](https://wandb.ai/). Use one W&B project per course project; group members share it. For every run, log:

- **Configuration:**
  - architecture
  - optimizer
  - learning rate and schedule
  - batch size
  - number of epochs
  - regularization settings
  - random seed
- **Training curves:** training and validation loss and metrics for every epoch, and the learning rate.
- **Final results:** test metrics of your final models.
- **Diagnostics:** e.g. sample predictions, confusion matrices or ROC curves.

Name, tag or group your runs, so that baselines, hyperparameter searches, ablations and final models are easy to tell apart. Results and figures in your report should come from runs logged in this project. If you use W&B sweeps for hyperparameter search, include them too.

> ⚠️ **Important:** Do **not** add the lecturer to your W&B project. Instead, make the project **public** (*Project page → Overview → Edit → Project Visibility → Public*) and provide its link in your report and in the Moodle submission comments (Section 4).

---

## 7. Project Presentation

Each group gets a **30-minute slot**: **15 minutes for the presentation**, in which every member presents their own part, and **15 minutes for questions**. In the question part, **every student gets at least three questions** about the project. The questions aim at your **theoretical understanding of the methods you used**.

Present the main points of your work:
- the learning task,
- the data analysis,
- the literature overview,
- your approach and its implementation,
- the results and their analysis.

You can prepare and give your presentation with any tool you like, e.g. PowerPoint, LaTeX Beamer, or the Jupyter notebook of your project shown as a slideshow with the RISE plug-in. Upload your slides to Moodle as a **PDF**.

### What we look for

The presentation and your answers to the questions are assessed on:
- **Learning task and data:** a clear formulation of the task, the properties of the data, and splits without data leakage.
- **Model choices:** why your architecture, output activation and loss function fit the task.
- **Training:**
  - your choice of optimizer, learning rate and batch size;
  - how you detected convergence problems and overfitting, e.g. from learning curves;
  - how you addressed them, e.g. with regularization, early stopping or data augmentation.
- **Evaluation:** suitable measures, a fair comparison with baselines, the reliability of your results, your ablation study, the comparative analysis of the group's approaches, and what your error analysis shows.
- **Understanding:** you can explain the mathematics behind the methods you use, e.g. your loss function, how gradients flow through your model, or how attention works, as well as the limitations of your results.
- **Presentation:** a clear structure, readable slides and keeping to the time.
