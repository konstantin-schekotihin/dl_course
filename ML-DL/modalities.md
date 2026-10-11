# Machine Learning & Deep Learning: Course Modalities

This document describes how the course is organized, how it is graded, and what you need to submit and when.

---

## 1. Course Organization & Attendance

- **Format:** lectures combined with in-class exercises, plus **8 practical sessions**.
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
| &nbsp;&nbsp;• Project proposal | 10% of project (7% of final) | Written plan of your project (Section 5). |
| &nbsp;&nbsp;• Project report | 30% of project (21% of final) | 6–8 page paper with code and data (Section 7). |
| &nbsp;&nbsp;• Project presentation | 60% of project (42% of final) | 15-minute presentation followed by 5 minutes of questions (Section 6). |

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
| Project report (PDF + code archive) | 20 January 2027 | 30 January 2027 |
| Presentation slides (PDF) | 30 January 2027 | 8 February 2027 |
| **Presentations** | — | 2nd week of February 2027 (8–12 Feb) |

Registration for a presentation slot will be done using the registration link provided in Moodle. **Presentations are in person.**

---

## 4. Projects

A project shows that you can apply machine learning methods to a problem of your choice. Suitable domains include computer vision, natural language processing, time series analysis and reinforcement learning. A project has three deliverables: (1) proposal, (2) report, and (3) presentation.

### Finding a topic

Start with a domain that interests you, for example computer vision. Then identify a task in that domain, such as image classification, object detection, image generation or image modification. These websites list tasks, datasets and approaches to solve them:
- ScholarWiki or SOTAPapers (PapersWithCode mirrors)
- [Kaggle](https://www.kaggle.com/)
- [Hugging Face](https://huggingface.co/)
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
- [Google Scholar](https://scholar.google.com/) or [Semantic Scholar](https://www.semanticscholar.org/)

On paper-listing sites, solutions are described in scientific publications. On Kaggle, they are described in notebooks written by professionals, students and enthusiasts. You may use any material you find on the Web. The only requirement is that **you understand and can explain the approach, the data and the results**.

> 📝 **Example:** A student interested in astronomy picks the pulsar classification problem. Given measurements from radio telescopes, the task is to classify stars into pulsars (1) and non-pulsars (0). Kaggle lists solutions to this problem as notebooks. A typical solution:
> - visualizes the data and studies the statistical properties of the features, to find the most valuable features and to get a first idea of which ML methods can train a classifier;
> - then trains classification models and compares them using measures such as accuracy, precision and recall.

Excellent deliverables of students from previous years are available in the **Examples** folder in Moodle. Feel free to use them as a guide.

Please get in touch with the lecturer if you have any questions!

### Group projects

A group of students can work on one project. This is reserved for complex projects in which every student does the same amount of work as in a single-person project. The requirements:
- Discuss group projects with the lecturer **before** submitting the proposal.
- Each student provides their own solution to the project's learning task, using an approach that is significantly different from those of the other group members.
- Every deliverable states clearly what each student is expected to do and has done. For example, in the learning task section: "In this project, *(name)* will focus on … whereas *(name)* will …".
- The group submits one version of every deliverable. Every member must be able to answer any question about the presented work.

---

## 5. Project Proposal

Prepare your proposal with LaTeX or an office program and submit it as a **PDF**. It must contain at least the following five sections and answer all of their questions:

1. **Motivation:** Why are you doing this project, e.g. personal motivation or importance of the problem? What is the goal of your application, and why is it important? What is your hypothesis, i.e. why do you believe the problem is solvable?
2. **Learning task:**
   - *Training experience:* Where does your training data come from? Why do you think it can actually be used to solve the problem? Show examples of your data and use them to explain the learning task.
   - *Learning task:* What are you going to do? Which type of learning suits the task?
   - *Performance measure:* How will you measure success? Which measures will you use, and why? How do you interpret their values for your problem? For example, is a precision of 0.7 good or bad?
3. **Previous work:** Summarize two to three papers, blog posts or articles that consider a learning task (problem, dataset, measures) similar to yours. How can you apply their approach to your problem? What modifications might be required? Can you use it as a baseline in your evaluation?
4. **Plan:** How will you solve the learning task with (an extension of) state-of-the-art methods? Describe your approach in enough detail to show its benefits, how it differs from previous work, and that it can be implemented within the project.
5. **Risk management:** What is your "plan B" if something does not go as expected, e.g. no data, poor-quality data, or training that does not converge?

If you have doubts about your proposal, contact the lecturer well before the deadline!

---

## 6. Project Presentation

You get a 20-minute slot: 15 minutes for the presentation and 5 minutes for questions. Present the main points of your work:
- the learning task,
- the data analysis,
- the literature overview,
- your approach and its implementation,
- the results and their analysis.

Prepare the slides in PowerPoint or LaTeX Beamer and upload them to Moodle as a **PDF**. You can also present the Jupyter notebook of your project as a slideshow using the RISE plug-in.

---

## 7. Project Report

The report is a **6–8 page** document structured like a research paper. You may adapt the structure below, but it is a good starting point.

1. **Motivation:** the problem you are solving and why it is important, plus a short summary of your methods and results.
2. **Data:** the data you use, with examples.
   - Where does it come from?
   - What are its properties, such as class balance, missing values and scale, and how do you deal with them?
   - Which preprocessing, filtering, augmentation or other manipulations do you apply?
3. **Theory and approach:**
   - Literature overview: which models from the literature suit this problem and your data, and why? Which algorithms can train them? How do these models relate to your hypothesis?
   - Overview of your approach: which steps does your learning pipeline execute? Which hyperparameters does the learning algorithm have, and how can they influence the results?
   - You must show that you applied methods covered during the semester to your problem. Figures and tables that describe your method or compare it with others are welcome. Cite all information from other sources properly.
4. **Implementation:** how you implemented your approach and which tools you used. Show interesting code snippets or present your algorithms.
5. **Evaluation:** present and visualize your results and explain what they mean. Why was your approach successful, or why not? Compare it to a baseline, implemented by yourself or taken from the literature, blogs, Kaggle, etc.
6. **Conclusions:** how do you assess the results of your work, and what would you recommend as next steps?
7. **AI disclosure statement:** see Section 8.

### Submission

Submit **two files** in Moodle:
1. the report as a **PDF**;
2. **one archive** with all code and artifacts needed to reproduce and evaluate your work: data, custom Python modules, images, videos, etc. If the archive is too large, publish it elsewhere (e.g. on GitHub) and put the link in the report. The same applies to large datasets: provide a link from which they can be downloaded.

Paste the link to your **Weights & Biases (W&B) project** into the submission comments, because Moodle does not open links from the report review page. Do not invite the lecturer to the project. Instead, make it public (*Project page → Overview → Edit → Project Visibility → Public*) and share the link.

> ⚠️ **Important:** Reports that do not meet these requirements will not be considered!

---

## 8. Generative AI & AI Coding Assistants

You may use generative AI tools and coding assistants (e.g. ChatGPT, Claude, GitHub Copilot) for your project, under these conditions:

1. **Disclosure:** declare any use of AI tools in an *AI disclosure statement* at the end of your proposal and report. State which tools you used and for what purpose, e.g. brainstorming, literature search, code generation or text drafting.
2. **You must be able to explain everything you submit.** During the presentation and the questions, you must be able to explain and justify every part of your work: the data processing, model architecture, loss function, training procedure, hyperparameters and evaluation. An AI tool cannot answer the questions for you.
3. **Check what the tools produce.** You are responsible for the correctness of AI-generated code and text. Typical errors to look for:
   - **data leakage**, e.g. normalization statistics computed on the whole dataset before the train/test split, or test data used for model selection;
   - **unsuitable evaluation**, e.g. accuracy on a heavily imbalanced dataset instead of precision/recall, $F_1$ or ROC-AUC;
   - **mismatched losses and outputs**, e.g. applying softmax before a loss that already includes it, such as `torch.nn.CrossEntropyLoss`;
   - **invented APIs**, i.e. functions or parameters that do not exist in the library version you use;
   - **references** that do not exist or do not say what is claimed.

---

## 9. Recommended Literature

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
