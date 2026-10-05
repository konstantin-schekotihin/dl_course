# Question Bank: Preparatory
Here is a collection of questions related to prep that can be taken for the examples in class, questions for homework or even the exams.

TODO: The questions are given a perceived level of difficulty and the corresponding answers are in the `answers_preparatory.md` file.

**Question 1**

An agentic AI system writes code blocks using two models, Model A and Model B. A code block is selected at random.

- Model A is used 60% of the time and has a 5% chance of producing a code block that contains an error.
- Model B is used 40% of the time and has a 2% chance of producing a code block that contains an error.

(1) Find the probability that the selected code block contains an error.

(2) Given that the selected code block contains an error, find the probability that it was produced by Model A.

(3) Find the probability that a code block is either produced by Model A or contains an error.

---

**Question 2**

A bank uses an automated system to flag potentially fraudulent transactions. Only 1% of all transactions are actually fraudulent.

Two detection systems are used:

- **System A (Behavioural Model)**
  - Detects fraud correctly with probability 0.95.
  - Incorrectly flags a legitimate transaction with probability 0.08.
- **System B (Rule-based Model)**
  - Detects fraud correctly with probability 0.85.
  - Incorrectly flags a legitimate transaction with probability 0.03.

Assume the systems are **independent given whether the transaction is fraudulent or not**.

(1) Find the probability that a randomly selected transaction is flagged by both systems.

(2) Given that a transaction is flagged by **both systems**, find the probability that it is actually fraudulent.

(3) A transaction is flagged by **System A only**. Find the probability that it is actually fraudulent.

(4) Explain why the assumption of independence is important in this problem, and what might happen if the systems were not independent.

(5) **Optional Challenge:** The bank adds a third AI model with:

- True positive rate = 0.90
- False positive rate = 0.02

If all three systems flag a transaction, should the bank automatically block it? Justify your answer quantitatively.

---

**Question 3**

Use the three axioms of probability (Kolmogorov's axioms) to prove the following statements for arbitrary events $\mathcal{A}$ and $\mathcal{B}$.

(1) **Complement rule:**

$$
\mathbb{P}[\mathcal{A}^c] = 1 - \mathbb{P}[\mathcal{A}]
$$

(2) **Impossible event:**

$$
\mathbb{P}[\emptyset] = 0
$$

(3) If $\mathcal{A}$ is contained in $\mathcal{B}$, i.e. $\mathcal{A} \subseteq \mathcal{B}$, prove that

$$
\mathbb{P}[\mathcal{A}] \leq \mathbb{P}[\mathcal{B}]
$$

(4) **General addition rule:**

$$
\mathbb{P}[\mathcal{A} \cup \mathcal{B}]
=
\mathbb{P}[\mathcal{A}]
+
\mathbb{P}[\mathcal{B}]
-
\mathbb{P}[\mathcal{A} \cap \mathcal{B}]
$$

---

**Question 4**

A delivery company tracks daily performance:

- $X$: number of deliveries completed (in hundreds)
- $Y$: total delay time due to traffic (in hours)

The random variables satisfy:

$$
\mathbb{E}[X] = 8, \qquad \mathbb{E}[Y] = 3
$$

$$
\operatorname{Var}(X) = 4, \qquad \operatorname{Var}(Y) = 2
$$

$$
\operatorname{Cov}(X,Y) = -1.5
$$

(1) Find the expected value of

$$
Z = 5X - 2Y
$$

(2) Find the variance of $Z$.

(3) Interpret the negative covariance between $X$ and $Y$ in this context.

(4) The company defines efficiency as

$$
E = X-Y
$$

Would reducing the variability of $Y$ necessarily reduce the variability of $E$? Justify your answer quantitatively.

(5) **Optional Challenge:** If $X$ and $Y$ were independent, how would your answer to part (2) change?

---

**Question 5**

Consider the scalar function

$$
\phi = xe^{-y^2}
$$

(1) Find

$$
\frac{\partial \phi}{\partial x}
\qquad \text{and} \qquad
\frac{\partial \phi}{\partial y}
$$

Confirm that

$$
\frac{\partial^2 \phi}{\partial x \partial y}
=
\frac{\partial^2 \phi}{\partial y \partial x}
$$

(2) If

$$
y = \sqrt{1-x^2}
$$

calculate the total derivative

$$
\frac{d\phi}{dx}
$$