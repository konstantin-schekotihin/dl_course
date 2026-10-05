# Extended & Specialized Modules

**Repository Path:** `Extended/`

This directory houses specialized elective modules, advanced extensions, and domain-specific topics that build upon the core machine learning and deep learning foundations.

---

## 📚 Available Modules

### `01_reinforcement_learning/` — Reinforcement Learning: Multi-Armed Bandits
- **Lecture Notebook:** [`07_RL.ipynb`](01_reinforcement_learning/07_RL.ipynb)
- **Supporting Simulation Package:** `rf/` (Bandit agents, stationary/non-stationary environments, reward policies)
- **Topics Covered:**
  - The Exploration vs. Exploitation dilemma in sequential decision making under uncertainty.
  - Action-value methods, sample-average estimates, and incremental update rules.
  - $\epsilon$-greedy action selection and tracking non-stationary problems.
  - Optimistic initial values for initial exploration.
  - Upper-Confidence-Bound (UCB) action selection using Hoeffding's inequality.
  - Thompson Sampling (Bayesian posterior updating for Bernoulli/Gaussian bandits).
  - Introduction to Deep Q-Networks (DDQN) and Markov Decision Processes (MDPs).

---

## 🚀 Execution & Usage

```bash
# Ensure project dependencies are synced
uv sync

# Launch Jupyter Notebook
uv run jupyter notebook
```
Navigate to `Extended/01_reinforcement_learning/07_RL.ipynb`. Interactive RISE slides can be activated via **`Alt + R`**.
