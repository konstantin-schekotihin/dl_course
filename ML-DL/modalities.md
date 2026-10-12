# Machine Learning & Deep Learning: Course Modalities

This document describes how the course is organized, how it is graded, and what you need to submit and when.

## Course goals

The course focuses on deep learning. It starts with a short recap of the mathematical foundations (probability, information theory, linear algebra), which are usually covered in other courses. Most of the time, including most exercises, is spent on deep learning itself.

After completing the course, you will be able to:
- formulate a learning task with its training experience, task and performance measure, and choose suitable data and evaluation measures;
- design, implement and train neural networks in PyTorch, from single-layer models to convolutional networks, sequence models, transformers and graph neural networks;
- choose an architecture, loss function and optimizer for a given task, and justify the choice;
- diagnose and fix training problems such as overfitting, vanishing gradients and poor convergence, using regularization and modern optimizers;
- evaluate models rigorously, with ablation studies and comparisons against the state of the art, and report the results reproducibly in the form of a scientific paper.

---

## 1. Course Organization & Attendance

- **Format:** lectures combined with in-class exercises, plus **8 practical sessions** with homework submissions and a written exam at the end of the semester (Section 2). The schedule and tasks of the practical sessions are announced by the lecturer of the practical sessions.
- **Attendance:** not compulsory, but strongly recommended. If you miss a class, you don't need an excuse, but you are responsible for the material covered and all announcements made in class.
- **Online resources** (links in Moodle):
  - Course notebooks: <https://github.com/konstantin-schekotihin/dl_course>
  - Announcements and feedback: Moodle forum
- **DataCamp:** you can register for DataCamp as complementary self-study. It is not part of the grading.

---

## 2. Assessment & Grading

| Component | Weight | Description |
| :--- | :---: | :--- |
| **Practical sessions** | **30%** | Homework submissions and a written exam, at most 100 points (see below). |
| **Project** | **70%** | Made up of three deliverables: |
| &nbsp;&nbsp;• Project proposal | 10% of project (7% of final) | Project plan, up to 4 pages (Section 4). |
| &nbsp;&nbsp;• Project report | 30% of project (21% of final) | Research paper, up to 9 pages, with code and W&B logs (Section 4). |
| &nbsp;&nbsp;• Project presentation | 60% of project (42% of final) | 30-minute slot per group: 15 minutes of presentation, 15 minutes of questions, at least 3 to every student (Section 4). |

### Practical sessions

The practical part is graded with **at most 100 points**, made up of:
- **Homework submissions (up to 16 points):** you submit your solution of each of the 8 practical sessions in Moodle. Each submission earns
  - **0 points** if less than half of it is solved,
  - **1 point** if the majority is solved, but with mistakes,
  - **2 points** if everything is solved, with at most small issues.
- **Written exam (up to 100 points):** one written exam at the end of the semester, with pen and paper, lasting **90 minutes**. It contains **4 questions**, of which you choose **3** and solve them. The exam date will be announced in Moodle as soon as the room is booked.

The points add up, but the total is capped at 100:

$$\text{Practicals} = \min(100,\ \text{Exam} + \text{Homework})$$

For example, 80 exam points and 16 homework points give 96 points. 90 exam points and 12 homework points give 100 points, because of the cap.

### Final grade

The practical sessions and each project deliverable are graded with 0–100 points. The scores combine as follows:

$$\text{Project} = 0.10 \cdot \text{Proposal} + 0.30 \cdot \text{Report} + 0.60 \cdot \text{Presentation}$$

$$\text{Final score} = 0.30 \cdot \text{Practicals} + 0.70 \cdot \text{Project}$$

> ⚠️ **Important:** To complete the course with a positive grade, you must collect **at least 60 of the 100 points** in the practical sessions **and** reach a final score of at least 60.

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
| Project proposal (PDF) | 10 November 2026, 23:59 | 15 November 2026, 23:59 |
| Project report (PDF + code) | 20 January 2027, 23:59 | 30 January 2027, 23:59 |
| Presentation slides (PDF) | 30 January 2027, 23:59 | 8 February 2027, 08:00 |
| **Presentations** | — | 2nd week of February 2027 (8–12 Feb) |

All times are local time in Klagenfurt (CET).

Registration for a presentation slot will be done using the registration link provided in Moodle. **Presentations are in person.**

---

## 4. Projects

Projects are **group work for 2–3 students**. You are **free to choose** your topic, your data, the deep learning methods you develop, and how you divide the work within the group. Suitable domains include computer vision, natural language processing, time series analysis and reinforcement learning. A project has three deliverables: (1) proposal, (2) report, and (3) presentation.

> ⚠️ **Important:** Whatever you build, every project **must** contain:
> 1. an **ablation study**: which components of your approach actually contribute to the result? Remove or replace one component at a time and compare.
> 2. a **comparative analysis** of your results: compare the models and variants you trained with each other, and with the **state of the art**, i.e. the best published results on the same data.
> 3. **W&B tracking of all experiments**, including the ablation runs, in a public Weights & Biases project.

The **[project guide](https://github.com/konstantin-schekotihin/dl_course/tree/master/ML-DL/deliverables)** describes in detail how to find a topic and how to prepare each deliverable. It also contains the LaTeX templates for the proposal and the report.

### Group work

Projects are done in **groups of 2–3 students**. Use the Moodle Group Choice tool to find teammates.
- **Clear attribution:** every deliverable states what each student is expected to do and has done. For example: "In this project, *(name)* will focus on … whereas *(name)* will …".
- **One deliverable per group:** the group submits one proposal and one report. Every member must be able to answer any question about the presented work.

**Individual assessment:**
- The proposal and the report are graded once, for the whole group.
- In the presentation, every member presents their own part. Each member is then questioned individually, both on their own part and on the project as a whole.
- The presentation is graded individually for each member.

### Deliverables at a glance

- **LaTeX templates:** the course provides LaTeX templates for the proposal and the report. Open them in [Overleaf](https://www.overleaf.com/) with one click from the [project guide](https://github.com/konstantin-schekotihin/dl_course/tree/master/ML-DL/deliverables), and share the project with your group through its edit link (link sharing) to write together. You don't need to install LaTeX.
- **Proposal:** at most **4 pages** (excluding references), written with the course LaTeX template and submitted as a PDF. It includes your plan for the ablation study and the comparative analysis, including the state-of-the-art results you will compare against.
- **Report:** at most **9 pages** (excluding references), written with the course LaTeX template and submitted as a PDF. It **must** contain the **ablation study** and the **comparative analysis**, within your study and against the state of the art. Also submit:
  - **your code**, as an archive or repository, or as a Google Colab notebook;
  - **a link to a public Weights & Biases (W&B) project** with **all** your experiments, including the ablation runs. Do **not** add the lecturer to the project; make it public and provide the link.

  Reports that do not meet the requirements in the project guide will not be considered.
- **Presentation:** a **30-minute slot** per group, with **15 minutes for the presentation** and **15 minutes for questions**. Every member presents their own part, and every student gets **at least three questions** on the theoretical understanding of the methods used. You can present with any tool you like; upload your slides to Moodle as a PDF.

Please get in touch with the lecturer if you have any questions!

---

## 5. Generative AI & AI Coding Assistants

You may use generative AI tools and coding assistants (e.g. ChatGPT, Claude, GitHub Copilot) for your project, under these conditions:

1. **Disclosure:** declare any use of AI tools in an *AI disclosure statement* in the appendix of your proposal and report. State which tools you used and for what purpose, e.g. brainstorming, literature search, code generation or text drafting.
2. **You must be able to explain everything you submit.** During the presentation and the questions, you must be able to explain and justify every part of your work: the data processing, model architecture, loss function, training procedure, hyperparameters and evaluation. An AI tool cannot answer the questions for you.

---

## 6. Recommended Literature

### Course books
- Bishop, C. M., & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. [Book website](https://www.bishopbook.com/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Book website](https://www.deeplearningbook.org/)
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. [Free online version](https://d2l.ai/)

### Introductory machine learning books
These books give a good and natural introduction to the main ML concepts. The first two are rather old, but still very readable.
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
- Duda, R. O., Hart, P. E., & Stork, D. G. (2000). *Pattern Classification* (2nd ed.). Wiley.
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer. [Free PDF](https://www.statlearning.com/) — a very clearly written book for beginners.

### Machine learning classics
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Free PDF](https://hastie.su.domains/ElemStatLearn/)

### Python for machine learning
- Raschka, S., Liu, Y., & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing.
