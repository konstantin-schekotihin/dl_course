# Exercise Set 1 - Linear Regression Draft

> **complete all**

**Motivation:**
The aim for this set is to grasp the simple aspects of Linear Regression in 1D and then extend this concept to N-D.

## Worded Questions

**Question 1**

What tasks would you solve with linear regression? When would it not be suitable?



## Analytical Exercises


**Question 1**

You are investigating the relationship between age and the amount of exercise people do. You take a survey of 10 people at a local park and record their age and the amount of hours they exercise for per week. The results are as follows:

| Age | Hours of Exercise Per Week |
| --- | -------------------------- |
| 22  | 7.8 |
| 28  | 6.5 |
| 31  | 7.1 |
| 35  | 5.9 |
| 40  | 5.2 |
| 45  | 4.6 |
| 50  | 4.9 |
| 55  | 3.4 |
| 62  | 3.0 |
| 68  | 2.1 |

a. Fit a Linear Regression model to this data **on paper**, where y is the Hours of Exercise and x is the age. You should state your answer in the form y = mx + c. Be ready to explain the full method in class.

b. Using this model predict the expected number of hours of exercise a week an 80 year old may partake in. Why is this not a reasonable prediction to make?

c. Why may this model be not hold up in practice?


**Question 2***

Let $X \in \mathbb{R}^{n \times (d + 1)}$ be the input matrix (n samples, d features + a bias term), $\mathbf{y} \in \mathbb{R}^n$ be the output vector and $\mathbf{\theta} \in \mathbb{R}^{d+1}$ be the weight vector.

a. Derive the OLS solution that minimises the residual sum of squares $||\mathbf{y} - \mathbf{\theta}X ||^2$.

b. Under what conditions does this solution fail?

c. *(extension)* Name one practical fix for the above condition and explain why.


## Practical Exercises

In the `linear_regression.ipynb` notebook, you will find two practical exercises. **Complete them both.**

Submit your answers to the worded and analytical questions in a pdf file, along with the completed `linear_regression.ipynb` notebook.
Additionally, submit the two `.npy` files containing the predictions and the MSE.



\* denotes exam-level questions