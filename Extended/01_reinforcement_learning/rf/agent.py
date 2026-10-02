from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import os, time

class Decreasing:

    def step(self, n):
        return 1 / n
    
    def name(self):
        return "decreasing"

class Constant(Decreasing):

    def __init__(self, step_size=0.1):
        self._step_size=step_size
        
    def step(self, n):
        return self._step_size

    def name(self):
        return "constant"

class UnbiasedConstant(Constant):
    
    def __init__(self, step_size=0.1):
        self._last = 0
        super().__init__(step_size)

    def step(self, n):
        self._last += self._step_size*(1 - self._last)
        return self._step_size / self._last

    def name(self):
        return "unbiased constant"

class Stationary:
    """
    Class for generation of rewards from a stationary normal distribution.
    """
    def __init__(self, means, std=1):
        """Stationary distribution for each action is defined by its mean. The standard deviation is the same for all actions.
           The class always stores initial values of means and standard deviation. 
           `reset` method can be used to restore the initial values, if required. 

        Arguments:
            means {array of floats} -- defines means of rewards for each action
        
        Keyword Arguments:
            std {int} -- standard deviation of rewards for all actions (default: {1})
        """
        self._std = std
        self._means = np.array(means)
        self._means.flags.writeable = False
        self.reset()
        
    def get_reward(self, action_id, **kwargs) -> float:
        """Returns a reward for a given action sampled from the normal distribution with 
        the predefined mean and standard deviation. 
        
        Arguments:
            action_id {int} -- identifier of an action
        
        Returns:
            float -- value of the reward
        """
        return np.random.normal(self._cmeans[action_id], self._std)

    def name(self):
        return 'Stationary'

    def actions_count(self):
        return len(self._cmeans)
    
    def reset(self):
        self._cmeans = np.array(self._means)

class Nonstationary(Stationary):
    """Generator of rewards from a non-stationary normal distribution. That is, before generating a reward, the mean of the action distribution is altered by a value sampled from the normal distribution with 0 mean and `mod_std` standard deviation defined in the constructor. `reset` method can be used to reset modivied means to its initial values.   
    """
    def __init__(self, means, trials=0, std=1, mod_std=0.01):
        """
        The class supports two types of generation:
          - random, when means are altered in arbitrary way before a reward is generated, `trials=0`
          - stable, when alterations of means are fixed over multiple runs, `trials>0`

        Arguments:
            means {array of floats} -- defines means of rewards for each action
        
        Keyword Arguments:
            trials {int} -- number of expected trials. Any value `<1` corresponds to the random generation (default: {0})
            std {int} -- standard deviation of rewards for all actions (default: {1})
            mod_std {float} -- standard deviation of a normal distribution used to sample changes of initial means (default: {0.01})
        """
        super().__init__(means, std)
        self._mod_std=mod_std
        if trials < 1:
            self._trials = None
        else:
            self._trials = np.random.normal(0, mod_std, trials)
            self._ids = np.random.randint(0, len(means), trials)
    
    def get_reward(self, action_id, trial):
        if self._trials is None:
            self._cmeans[action_id] += np.random.normal(0, self._mod_std)
        else:
            mod_id = self._ids[trial]
            self._cmeans[mod_id] += self._trials[trial]
        return super().get_reward(action_id)

    def name(self):
        return 'Nonstationary'
    
    def reset(self):
        super().reset()

class Bandit:

    def __init__(self, generator : Stationary, step : Decreasing, init_value=0.001):
        """Initialization of the single state
        
        Arguments:
            generator {Stationary} -- generator of rewards
            step {Decreasing} -- stepping strategy
        
        Keyword Arguments:
            init_value {float} -- initial value of actions in the table (default: {0.001})
        """
        self._generator = generator
        self._step = step
        self._init_value = init_value
        
    def _session(self, eps, trials):
        # reset action rewards and the generator for the new iteration
        action_rewards = []
        self._generator.reset()
        # create a list of actions
        actions = SortedList([[i, self._init_value, 0] for i in range(self._generator.actions_count())], key=lambda tup: -1 * tup[1])

        for i in range(trials):
            rnd = 0
            # greedy selection of an action
            if np.random.random() < eps:
                rnd = np.random.randint(0, self._generator.actions_count())
            a_id, old_reward, calls = actions.pop(rnd)

            # increase number of the action calls
            calls += 1

            # call the action (pull the lever)
            current_reward = self._generator.get_reward(a_id, trial=i)
            action_rewards.append(current_reward)

            # update rule
            step = self._step.step(calls)
            actions.add((a_id, old_reward + step * (current_reward - old_reward), calls))

        return np.array(actions), np.array(action_rewards)

    def _plot_results(self, results, show_last_run=True, show_avg_actions=True):
        runs = len(results)
        a_count = len(results[0][0])
        trials = len(results[0][1])

        cumulative_rewards = np.zeros(trials)
        cumulative_calls = np.zeros(a_count)
        actions = []
        action_rewards = []

        for run in range(runs):
            actions = results[run][0]
            action_rewards = results[run][1]

            # update rewards and action calls statistics
            for i in range(trials):
                avg = cumulative_rewards[i]
                cumulative_rewards[i] = (run * avg + action_rewards[i]) / (run + 1)

            for i in range(a_count):
                avg = cumulative_calls[i]
                calls = [item for item in actions if i == item[0]][0][2]
                cumulative_calls[i] = (run * avg + calls) / (run +1)

        # plot the results
        if show_avg_actions:
            fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
            ax = axs[0]
        else:
            fig, ax = plt.subplots(1,1)
        
        fig.set_figheight(5)
        fig.set_figwidth(15)
        fig.suptitle("{} rewards distribution with {} step".format(self._generator.name(), self._step.name()), fontsize=12)

        ax.scatter(range(trials), cumulative_rewards, s=12, zorder=0)
        avg = np.average(cumulative_rewards)
        ax.plot([0, trials],[avg, avg], color='red', zorder=1)
        ax.text(0, avg+0.05*avg, "avg. reward = {}".format(round(avg,4)), color='red', zorder=1, fontsize=14)
        ax.plot(cumulative_rewards, zorder=0)
        ax.set_title("Avg. rewards over all runs")
        if show_avg_actions:
            axs[1].bar(range(1, 11), cumulative_calls)
            axs[1].set_xticks(range(1, 11))
            axs[1].set_title("Avg. number of calls")
        plt.show()
        
        if show_last_run:
            fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
            fig.set_figheight(5)
            fig.set_figwidth(15)
            axs[0].plot(action_rewards)
            axs[0].set_title("Rewards during the last run")
            axs[1].bar(actions[:, 0] + 1, actions[:, 2])
            axs[1].set_xticks(range(1, 11))
            axs[1].set_title("Number of calls per action")
            plt.show()

    def run_parallel(self, trials=1000, runs=2000, eps=0.1, **kwargs):
        start_time = time.time()
        with mp.Pool(processes=os.cpu_count() - 2) as pool:
            results = [
                pool.apply(self._session, args=(eps, trials)) for run in range(0, runs)]

        print("Execution time: %s seconds" % (time.time() - start_time))   
        self._plot_results(results, **kwargs)

    def run(self, trials=1000, runs=2000, eps=0.1, **kwargs):
        start_time = time.time()
        results = [self._session(eps, trials) for run in range(0, runs)]
        print("Execution time: %s seconds" % (time.time() - start_time))
        self._plot_results(results, **kwargs)

if __name__ == '__main__':
    np.random.seed(19801031)
    means = [0.2, -0.7, 1.5, 0.5, 1.2, -1.2, -0.2, -0.9, 0.7, -0.5]
    trials=100

    b = Bandit(Nonstationary(means, trials), UnbiasedConstant(0.2))
    b.run(trials, runs=2000, show_last_run=False, show_avg_actions=False)