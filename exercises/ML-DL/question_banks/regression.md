# Question Bank: Regression
Here is a collection of questions related to regression that can be taken for the examples in class, questions for homework or even the exams.

TODO: The questions are given a perceived level of difficulty and the corresponding answers are in the `answers_regression.md` file.

**Question 1**: Difficulty (1/10)

What is regression in the context of machine learning?

---

**Question 2**: Difficulty (1/10)

How does regression relate to artificial neural networks? And when are they the same?

---

**Question 3**: Difficulty (1/10)

Explain the difference between linear and non-linear regression.

---

**Question 4**: Difficulty (1/10)

Is non-linear regression still a linear model?

---

**Question 5**: Difficulty (1/10)

What are some common evaluation metrics for regression models?

---

**Question 6**: Difficulty (1/10)

Explain the concept of overfitting in regression models.

---

**Question 7**: Difficulty (1/10)

What are some common techniques to prevent overfitting in regression models?

---

**Question 8**: Difficulty (3/10)

Given the 1D regression equation $y = mx + c$, derive a solution for the minimum using a the loss function, $L = ||y - \hat{y}||_2$.

---

**Question 9**: Difficulty (.../10) [ANSWER CAN BE FOUND ONLINE]

For the logistic regression function, $\phi(z)$, show that
$$\frac{\partial}{\partial{z}}\phi(z) = \phi(z)(1-\phi(z))$$
How does the nature of the activation function affect the computational cost of the error-back propagation algorithm?

---

**Question 10**:

We would like to predict the price of a house based on its size (in square metres) and the number of bedrooms. We have a dataset of $N$ houses, where we have the size 
$x$, the number of bedrooms 
$z$, and the price 
$y$. We want to fit a linear regression model of the form:

$$y = w_0 + w_1 x + w_2 z$$

Where 
$w_0$ is the intercept, 
$w_1$ is the coefficient for the size, and 
$w_2$ is the coefficient for the number of bedrooms.

 Derive the closed-form solution for the parameters 
$w_0$, $w_1$, $w_2$. Provide each step of the derivation.

---

**Question 11**

I would like to model the relationship between the air quality index (AQI), temperature, and the number of hospital admissions in a given city.

Suppose we have a dataset of $N$ days. For each day $i = 1,\ldots,N$, we observe:

- the air quality index $a_i \in \mathbb{R}$,
- the temperature $t_i \in \mathbb{R}$,
- the number of hospital admissions $y_i \in \mathbb{R}$.

Thus, our dataset can be written as

$$
\mathcal{D} = \{(a_i, t_i, y_i)\}_{i=1}^{N}.
$$

Our aim is to predict the number of hospital admissions $y_i$ from the AQI $a_i$ and temperature $t_i$ using a regression model.


(1) Explain why a standard linear model

$$
\hat{y}_i = w_0 + w_1 a_i + w_2 t_i
$$

may not be sufficient to model the relationship between AQI, temperature, and hospital admissions.

(2) Define a suitable feature mapping

$$
\boldsymbol{\phi}(a_i,t_i)
$$

that includes non-linear effects of AQI and temperature. You may also consider interactions between the two variables.

(3) Using your feature mapping, write down the resulting regression model and show that, although the model may be non-linear in the original inputs $a_i$ and $t_i$, it is still linear in the model parameters $\mathbf{w}$.

---

**Question 12**

A student fits two regression models to a housing dataset. The dataset contains two input features, $x_1$ and $x_2$, and a target variable $y$ representing the house price.

- **Model A:** Uses the original features $x_1$ and $x_2$ in a standard linear regression model.
- **Model B:** Uses polynomial basis functions up to degree 5 for both input features.

After fitting both models to the same training data, Model B achieves a much lower training error than Model A.


(1) Explain why you would expect Model B to have a lower training error than Model A.

(2) Explain why Model B might nevertheless perform worse than Model A on unseen data.

(3) Describe one method that could be used to select between Model A and Model B in practice.

---