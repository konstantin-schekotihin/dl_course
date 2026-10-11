# Extended & Specialized Modules

**Repository Path:** `Extended/`

This directory houses specialized elective modules, advanced extensions, and domain-specific topics that build upon the core machine learning and deep learning foundations.

---

## 📚 Available Modules

### `01_reinforcement_learning/` — Reinforcement Learning: From Bandits to Deep Q-Networks
- **Lecture Notebook:** [`07_RL.ipynb`](01_reinforcement_learning/07_RL.ipynb)
- **Supporting Simulation Package:** `rf/` (bandit agents with stationary and decreasing step-size strategies)
- **Topics Covered:**
  - The learning agent and its environment; examples from Backgammon and Go.
  - The $k$-armed bandit problem: action values, sample-average estimates, and incremental update rules.
  - $\epsilon$-greedy action selection and optimistic initial values.
  - Non-stationary bandits: tracking changes with constant and unbiased constant step sizes; parameter studies.
  - Upper-Confidence-Bound (UCB) action selection and gradient bandit algorithms.
  - Value functions, on-policy SARSA, and off-policy Q-learning.
  - Deep Q-Networks (DQN) and an outlook on policy optimization.

---

## 🚀 Execution & Usage

```bash
# Ensure project dependencies are synced
uv sync

# Launch Jupyter Notebook
uv run jupyter notebook Extended/01_reinforcement_learning/
```

Navigate to `Extended/01_reinforcement_learning/07_RL.ipynb`. Interactive RISE slides can be activated via **`Alt + R`**.
The course slide stylesheet is kept in `shared/styles/rise.css`. RISE picks up a `rise.css` placed in the notebook's own folder.

> 📝 **Note:** The notebook imports its simulation package from the local `rf/` folder, so it currently needs a local checkout of the repository (it does not run when opened directly from GitHub in Colab).
