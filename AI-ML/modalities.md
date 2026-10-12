# AI and Machine Learning: Course Modalities

Welcome to the **AI and Machine Learning** course. It equips students from diverse disciplines with analytical and applied skills: to formulate learning tasks, select suitable algorithms, build leak-free pipelines, train and validate models, and draw defensible conclusions in their application domain.

This document describes how the course is organized, how it is graded, and what you need to submit and when.

---

## 1. Course Organization & Attendance

- **Format:** lectures combined with in-class interactive coding exercises and Jupyter notebook walkthroughs.
- **Attendance:** not compulsory, but regular participation is strongly recommended. If you miss a class, you don't need an excuse, but you are responsible for following the class discussions, coding exercises and organizational announcements.
- **Course notebooks:** <https://github.com/konstantin-schekotihin/dl_course> (folder `AI-ML`).

---

## 2. Assessment & Grading

| Component | Weight | Description |
| :--- | :---: | :--- |
| **DataCamp assignments** | *Prerequisite* | 4 online courses on the e-learning platform. You must complete **all 4** to be eligible for a positive grade. |
| **Project proposal** | **10%** | Written specification of the domain problem, the learning task $(T, P, E)$ and the research plan (Section 4). |
| **Project report and code** | **30%** | Scientific paper of up to 9 pages with a reproducible code archive and a public W&B workspace (Section 4). |
| **Oral defensio** | **60%** | 10-minute team presentation followed by a 20-minute individual oral examination (Section 4). |

The final grade is computed on a scale of 0 to 100 points:

$$\text{Final score} = 0.10 \cdot \text{Proposal} + 0.30 \cdot \text{Report} + 0.60 \cdot \text{Defensio}$$

| Final score | Grade |
| :---: | :--- |
| $[90, 100]$ | 1 — Sehr gut (Excellent) |
| $[80, 90)$ | 2 — Gut (Good) |
| $[70, 80)$ | 3 — Befriedigend (Satisfactory) |
| $[60, 70)$ | 4 — Genügend (Sufficient) |
| $[0, 60)$ | 5 — Nicht genügend (Fail) |

---

## 3. Submissions & Deadlines

All deliverables are submitted **only in Moodle**, using the corresponding upload task.

- **Due date:** by this date you must upload at least a draft, to register for the evaluation of the deliverable.
- **Draft state:** you don't have to click *Submit*; your submission stays editable as a draft until the cut-off date.
- **Cut-off date:** no updates are possible after this date. All submissions are locked and graded after the cut-off date.

The deadlines are the latest possible dates. Earlier submissions are welcome!

| Deliverable | Due date (draft) | Cut-off date |
| :--- | :---: | :---: |
| Project proposal (PDF) | 10 November 2026, 23:59 | 15 November 2026, 23:59 |
| Project report (PDF + code) | 20 January 2027, 23:59 | 30 January 2027, 23:59 |
| Defensio slides (PDF) | 30 January 2027, 23:59 | 8 February 2027, 08:00 |
| **Oral defensio** | — | 2nd week of February 2027 (8–12 Feb) |

All times are local time in Klagenfurt (CET). The oral defensio takes place **in person**; the room is announced in Moodle.

> ℹ️ **International and exchange students:** online participation in the oral defensio is possible upon prior agreement with the lecturer.

---

## 4. Projects

The project is a team project. The **[project guide](https://github.com/konstantin-schekotihin/dl_course/tree/master/AI-ML/deliverables)** describes how to find a topic and how to prepare each deliverable. It also contains the LaTeX templates for the proposal and the report.

### Teams and individual methods

- **Teams:** projects are done in teams of **exactly 3 students**. Use the Moodle Group Choice tool to find teammates and agree on an application domain. Exceptions are possible only after discussion with the lecturer.
- **Common task:** the team shares one dataset and one formal learning task $(T, P, E)$, e.g. credit card fraud detection, customer churn or biological image classification.
- **Distinct model families:** to allow a fair individual assessment in the defensio, each team member formulates, trains, tunes and evaluates a **fundamentally distinct model family**:
  - *Student A:* a linear or regularized parametric model (logistic regression, ElasticNet, OLS);
  - *Student B:* a non-linear tree ensemble (random forest, gradient-boosted decision trees);
  - *Student C:* a support vector machine (RBF kernel) or a multi-layer perceptron (MLP).
- **Clear attribution:** every section of the proposal and the report states who is expected to do, or has done, what, e.g. "The logistic regression baseline was implemented and evaluated by *(name)* …".
- **One deliverable per team:** the team submits one joint proposal and one joint report. In the defensio, each member is examined individually on their own model family and on their understanding of the comparison of all three.

### Deliverables at a glance

- **LaTeX templates:** the course provides LaTeX templates for the proposal and the report. Open them in [Overleaf](https://www.overleaf.com/) with one click from the [project guide](https://github.com/konstantin-schekotihin/dl_course/tree/master/AI-ML/deliverables), and share the project with your team through its edit link (link sharing) to write together. You don't need to install LaTeX.
- **Proposal:** at most **4 pages** (excluding references), written with the course template and submitted as a PDF.
- **Report:** at most **9 pages** (excluding references), written with the course template and submitted as a PDF, together with your reproducible code and the public link to your W&B workspace (Section 6).
- **Oral defensio:** a 10-minute team presentation with 5–6 content slides, followed by a 20-minute oral examination with targeted questions to each student.

---

## 5. Generative AI & AI Coding Assistants

You may use modern AI coding assistants (e.g. Antigravity, GitHub Copilot, ChatGPT, Claude Code) for your project, under these conditions:

1. **Disclosure:** declare any use of AI tools for brainstorming, exploratory data analysis, code generation, text drafting or literature search in an *AI disclosure statement* in the appendix of your proposal and report. State the tools, the model versions and the general purpose of their use.
2. **Defensibility:** an AI agent cannot sit the oral examination for you. In the defensio, every student must independently explain and defend **every line of code, mathematical formula, preprocessing choice and evaluation metric** presented in the deliverables.

Lecture `08_ml_agents.ipynb` presents a specification-driven way of working with AI coding agents and the *5-point Defensio Readiness Checklist*.

---

## 6. Mandatory Tooling: Weights & Biases (W&B)

To ensure reproducible machine learning and transparent progress tracking:
- Every project group must maintain a dedicated project workspace on **Weights & Biases (W&B)**.
- **What to Log**:
  - Training and validation loss curves across iterations / epochs.
  - Hyperparameter sweep configurations (`n_estimators`, `learning_rate`, `C`, `max_depth`, `k`).
  - Comparative validation tables ($F_1$-score, ROC-AUC, accuracy, runtime).
  - Diagnostic artifacts (confusion matrix heatmaps, ROC curves, feature importance plots).
- **Submission Requirement**: The final project report must include a **public, shareable link** to the group's W&B workspace.

---

## 7. Recommended Literature

### Core course reference
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer. [Free PDF and code](https://www.statlearning.com/)

### Notation and neural networks
- Bishop, C. M., & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. [Book website](https://www.bishopbook.com/) — the reference for the mathematical notation and for neural networks.

### Applied machine learning in Python
- Raschka, S., Liu, Y., & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing.

### Classic machine learning foundations
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. Chapter 19, *Learning from Examples*: decision tree induction, pruning and ensembles.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Free PDF](https://hastie.su.domains/ElemStatLearn/)
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill. The classical formalization of learning tasks $(T, P, E)$.
- Duda, R. O., Hart, P. E., & Stork, D. G. (2000). *Pattern Classification* (2nd ed.). Wiley.
