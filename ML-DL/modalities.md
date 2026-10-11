# Machine Learning & Deep Learning: Course Modalities

This document describes how the course is organized, how it is graded, and what you need to submit and when.

### Course goals

The course focuses on deep learning. It starts with a short recap of the mathematical foundations (probability, information theory, linear algebra), which are usually covered in other courses. Most of the time, including most exercises, is spent on deep learning itself.

After completing the course, you will be able to:
- formulate a learning task with its training experience, task and performance measure, and choose suitable data and evaluation measures;
- design, implement and train neural networks in PyTorch, from single-layer models to convolutional networks, sequence models, transformers and graph neural networks;
- choose an architecture, loss function and optimizer for a given task, and justify the choice;
- diagnose and fix training problems such as overfitting, vanishing gradients and poor convergence, using regularization and modern optimizers;
- evaluate models rigorously against baselines and report the results reproducibly, in the form of a scientific paper.

---

## 1. Course Organization & Attendance

- **Format:** lectures combined with in-class exercises, plus **8 practical sessions**. The schedule, tasks, submission rules and points per practical session will be announced by the lecturer of the practical sessions.
- **Attendance:** not compulsory, but strongly recommended. If you miss a class, you don't need an excuse, but you are responsible for the material covered and all announcements made in class.
- **Online resources** (links in Moodle):
  - Course notebooks: <https://github.com/konstantin-schekotihin/dl_course>
  - Announcements and feedback: Moodle forum
- **DataCamp:** you can register for DataCamp as complementary self-study. It is not part of the grading.

---

## 2. Assessment & Grading

| Component | Weight | Description |
| :--- | :---: | :--- |
| **Practical sessions** | **30%** | 8 practical sessions, 100 points in total. |
| **Project** | **70%** | Made up of three deliverables: |
| &nbsp;&nbsp;• Project proposal | 10% of project (7% of final) | Project plan, up to 4 pages (Section 5). |
| &nbsp;&nbsp;• Project report | 30% of project (21% of final) | Research paper, up to 9 pages, with code and W&B logs (Sections 6–8). |
| &nbsp;&nbsp;• Project presentation | 60% of project (42% of final) | 30-minute slot per group: 15 minutes of presentation, 15 minutes of questions, at least 3 to every student (Section 9). |

The practical sessions and each project deliverable are graded with 0–100 points. The scores combine as follows:

$$\text{Project} = 0.10 \cdot \text{Proposal} + 0.30 \cdot \text{Report} + 0.60 \cdot \text{Presentation}$$

$$\text{Final score} = 0.30 \cdot \text{Practicals} + 0.70 \cdot \text{Project}$$

> ⚠️ **Important:** To complete the course with a positive grade, you must collect **more than 50 of the 100 points** in the practical sessions **and** reach a final score of at least 60.

| Final score | Grade |
| :---: | :--- |
| $[90, 100]$ | 1 — Sehr gut (Excellent) |
| $[80, 90)$ | 2 — Gut (Good) |
| $[70, 80)$ | 3 — Befriedigend (Satisfactory) |
| $[60, 70)$ | 4 — Genügend (Sufficient) |
| $[0, 60)$ | 5 — Nicht genügend (Fail) |

---

## 3. Submissions & Deadlines

All project deliverables are submitted **only in Moodle**, using the corresponding upload task.

- **Due date:** by this date you must upload at least a draft of your submission.
- **Draft state:** you don't have to click *Submit*; drafts are accepted.
- **Updates:** you can upload new versions until the cut-off date, or until you click *Submit*.
- **Cut-off date:** no updates are possible after this date, not even by email. All submissions are graded after the cut-off date.

The deadlines are the latest possible dates. Earlier submissions are welcome!

| Deliverable | Due date (draft) | Cut-off date |
| :--- | :---: | :---: |
| Project proposal (PDF) | 10 November 2026 | 15 November 2026 |
| Project report (PDF + code) | 20 January 2027 | 30 January 2027 |
| Presentation slides (PDF) | 30 January 2027 | 8 February 2027 |
| **Presentations** | — | 2nd week of February 2027 (8–12 Feb) |

Registration for a presentation slot will be done using the registration link provided in Moodle. **Presentations are in person.**

---

## 4. Projects

Projects are **group work for 2–3 students**. A project shows that you can apply deep learning methods to a problem of your choice. Suitable domains include computer vision, natural language processing, time series analysis and reinforcement learning. A project has three deliverables: (1) proposal, (2) report, and (3) presentation.

### Finding a topic

Start with a domain that interests you, for example computer vision. Then identify a task in that domain, such as image classification, object detection, image generation or image modification. These websites list tasks, datasets and approaches to solve them:
- [ScholarWiki](http://scholarwiki.ai) or [SOTAPapers](https://sotapapers.com/) (successors of Papers with Code): tasks, benchmarks and leaderboards with linked papers and code
- [Kaggle](https://www.kaggle.com/)
- [Hugging Face](https://huggingface.co/)
- [Google Scholar](https://scholar.google.com/) or [Semantic Scholar](https://www.semanticscholar.org/)

On paper-listing sites, solutions are described in scientific publications. On Kaggle, they are described in notebooks written by professionals, students and enthusiasts. You may use any material you find on the Web. The only requirement is that **you understand and can explain the approach, the data and the results**.

> 📝 **Example:** A group interested in medical imaging picks pneumonia detection from chest X-rays, using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset on Kaggle. Given an X-ray image, the task is to classify it as *normal* or *pneumonia*. Kaggle lists many solutions to this problem as notebooks. A typical solution:
> - looks at example images and the class balance, and decides on preprocessing and data augmentation;
> - trains a convolutional network from scratch and fine-tunes a network pre-trained on ImageNet;
> - compares the models using precision, recall and ROC-AUC, since a missed pneumonia costs more than a false alarm.
>
> In a group, each member could take one of these approaches, or a third one such as a vision transformer.

Excellent deliverables of students from previous years are available in the **Examples** folder in Moodle. Feel free to use them as a guide.

Please get in touch with the lecturer if you have any questions!

### Paper format

Write the proposal and the report like papers for a machine learning conference:
- Use the official **NeurIPS LaTeX template** on [Overleaf](https://www.overleaf.com/latex/templates?q=neurips). Choose the template for the latest NeurIPS conference, or upload the [official style files](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip) to a new Overleaf project.
- Load the style with the `preprint` option, e.g. `\usepackage[preprint]{neurips_2025}`, so that your names appear on the paper. Without it, the template produces an anonymous submission.
- **Page limits**, **excluding references**:
  - **proposal:** at most **4 pages**, like a short (workshop) paper;
  - **report:** at most **9 pages**, like a full NeurIPS paper.

  Everything except the references counts towards the limit: figures, tables, appendices and the AI disclosure statement. Do not change margins, font sizes or spacing to fit more content.
- The NeurIPS paper checklist is not required.

### Group work

Projects are done in **groups of 2–3 students**. Use the Moodle Group Choice tool to find teammates.
- **Common task:** the group shares one domain, one dataset and one learning task (training experience, task and performance measure).
- **Distinct approaches:** each member formulates, trains, tunes and evaluates a **significantly different deep learning approach** to this task. For example, for image classification:
  - *Student A:* a convolutional network trained from scratch;
  - *Student B:* a pre-trained network fine-tuned to the task (transfer learning);
  - *Student C:* a transformer-based model, such as a vision transformer.
- **Clear attribution:** every deliverable states what each student is expected to do and has done. For example: "In this project, *(name)* will focus on … whereas *(name)* will …".
- **One deliverable per group:** the group submits one proposal and one report, which compare the approaches of all members. Every member must be able to answer any question about the presented work.

**Individual assessment:**
- The proposal and the report are graded once, for the whole group.
- In the presentation, every member presents their own part. Each member is then questioned individually, both on their own part and on the project as a whole.
- The presentation is graded individually for each member.

---

## 5. Project Proposal

Write your proposal in the paper format (Section 4) and submit it as a **PDF** of at most 4 pages. It must contain at least the following five sections and answer all of their questions:

1. **Motivation:** Why are you doing this project, e.g. personal motivation or importance of the problem? What is the goal of your application, and why is it important? What is your hypothesis, i.e. why do you believe the problem is solvable?
2. **Learning task:**
   - *Training experience:* Where does your training data come from? Why do you think it can actually be used to solve the problem? Show examples of your data and use them to explain the learning task.
   - *Learning task:* What are you going to do? Which type of learning suits the task?
   - *Performance measure:* How will you measure success? Which measures will you use, and why? How do you interpret their values for your problem? For example, is a precision of 0.7 good or bad?
3. **Previous work:** Summarize two to three papers, blog posts or articles that consider a learning task (problem, dataset, measures) similar to yours. How can you apply their approach to your problem? What modifications might be required? Can you use it as a baseline in your evaluation?
4. **Plan:** How will you solve the learning task with (an extension of) state-of-the-art methods? Describe the approaches in enough detail to show their benefits, how they differ from previous work, and that they can be implemented within the project. State which approach each group member takes.
5. **Risk management:** What is your "plan B" if something does not go as expected, e.g. no data, poor-quality data, or training that does not converge?

If you have doubts about your proposal, contact the lecturer well before the deadline!

---

## 6. Project Report

The report is a research paper of at most **9 pages** in the paper format (Section 4). Use the structure of a typical machine learning paper:

1. **Abstract:** one paragraph summarizing the problem, your approach and your main result.
2. **Introduction:** the problem you are solving and why it is important, your hypothesis, what you contribute, and a short summary of your results.
3. **Related work:** which models and approaches from the literature suit this problem and your data, and why? How does your work relate to them? Cite all sources properly.
4. **Data:** the data you use, with examples.
   - Where does it come from?
   - What are its properties, such as class balance, missing values and scale, and how do you deal with them?
   - Which preprocessing, filtering, augmentation or other manipulations do you apply?
   - How do you split the data into training, validation and test sets, and how do you prevent information from the test set leaking into training?
5. **Method:** your approach and why it fits the problem.
   - Model architecture, output activation and loss function.
   - Training procedure: optimizer, learning-rate schedule, regularization.
   - Which hyperparameters does your approach have, and how can they influence the results?
   - How do these choices relate to your hypothesis?
   - You must show that you applied methods covered during the semester to your problem. Figures of your architecture or pipeline are welcome.
6. **Experimental setup:**
   - Baselines: at least one simple baseline, and one from the literature, blogs, Kaggle, etc., if available.
   - Evaluation measures and why they fit the task.
   - Hyperparameter search: search space, strategy, and the data used for model selection.
   - Implementation details: framework, hardware and training time.
7. **Results:** present your main results in one comparison table covering the approaches of all group members and the baselines, with learning curves where they help. If feasible, report the mean and standard deviation over several random seeds. Explain what your results mean: why was your approach successful, or why not?
8. **Analysis:**
   - *Ablation study:* which parts of your approach contribute to the result?
   - *Error analysis:* on which examples does your model fail, and why?
9. **Limitations and conclusions:** what are the limits of your results? How do you assess your work, and what would you recommend as next steps?
10. **References**
11. **Appendix (optional):** additional results, the full hyperparameter configuration, and interesting code snippets or algorithms. Put the **AI disclosure statement** here (Section 10).

### Submission

Submit in Moodle:
1. the report as a **PDF**;
2. **your code**, in one of the two forms described in Section 7: either an archive, or a link to your repository or Colab notebook in the report. If an archive is too large, publish it elsewhere (e.g. on GitHub) and put the link in the report.

Paste the link to your public **Weights & Biases (W&B) project** (Section 8) into the submission comments, because Moodle does not open links from the report review page.

> ⚠️ **Important:** Reports that do not meet these requirements will not be considered!

---

## 7. Code & Reproducibility

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

Your evaluation should run on Google Colab or a standard laptop. If training needs more resources, state the hardware you used and the training time in the README.

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

## 8. Experiment Tracking with Weights & Biases

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

> ⚠️ **Important:** Do **not** add the lecturer to your W&B project. Instead, make the project **public** (*Project page → Overview → Edit → Project Visibility → Public*) and provide its link in your report and in the Moodle submission comments (Section 6).

---

## 9. Project Presentation

Each group gets a **30-minute slot**: **15 minutes for the presentation**, in which every member presents their own part, and **15 minutes for questions**. In the question part, **every student gets at least three questions** about the project. The questions aim at your **theoretical understanding of the methods you used**.

Present the main points of your work:
- the learning task,
- the data analysis,
- the literature overview,
- your approach and its implementation,
- the results and their analysis.

Prepare the slides in PowerPoint or LaTeX Beamer and upload them to Moodle as a **PDF**. You can also present the Jupyter notebook of your project as a slideshow using the RISE plug-in.

### What we look for

The presentation and your answers to the questions are assessed on:
- **Learning task and data:** a clear formulation of the task, the properties of the data, and splits without data leakage.
- **Model choices:** why your architecture, output activation and loss function fit the task.
- **Training:**
  - your choice of optimizer, learning rate and batch size;
  - how you detected convergence problems and overfitting, e.g. from learning curves;
  - how you addressed them, e.g. with regularization, early stopping or data augmentation.
- **Evaluation:** suitable measures, a fair comparison with baselines, the reliability of your results, and what your error analysis shows.
- **Understanding:** you can explain the mathematics behind the methods you use, e.g. your loss function, how gradients flow through your model, or how attention works, as well as the limitations of your results.
- **Presentation:** a clear structure, readable slides and keeping to the time.

---

## 10. Generative AI & AI Coding Assistants

You may use generative AI tools and coding assistants (e.g. ChatGPT, Claude, GitHub Copilot) for your project, under these conditions:

1. **Disclosure:** declare any use of AI tools in an *AI disclosure statement* in the appendix of your proposal and report. State which tools you used and for what purpose, e.g. brainstorming, literature search, code generation or text drafting.
2. **You must be able to explain everything you submit.** During the presentation and the questions, you must be able to explain and justify every part of your work: the data processing, model architecture, loss function, training procedure, hyperparameters and evaluation. An AI tool cannot answer the questions for you.

---

## 11. Recommended Literature

### Course books
- Bishop, C. M., & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. [Book website](https://www.bishopbook.com/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Book website](https://www.deeplearningbook.org/)
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. [Free online version](https://d2l.ai/)

### Introductory machine learning books
These books are rather old, but they give a good and natural introduction to the main ML concepts.
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
- Duda, R. O., Hart, P. E., & Stork, D. G. (2000). *Pattern Classification* (2nd ed.). Wiley.
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer. [Free PDF](https://www.statlearning.com/) — a very clearly written book for beginners.

### Machine learning classics
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Free PDF](https://hastie.su.domains/ElemStatLearn/)

### Python for machine learning
- Raschka, S., Liu, Y., & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing.
