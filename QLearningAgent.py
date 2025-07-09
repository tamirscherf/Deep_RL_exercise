import numpy as np
import torch
import torch.nn.functional as F
from utils import random_argmax, device
from Env import RestlessMultiArmedBandit
N_ARMS = RestlessMultiArmedBandit.N_ARMS
N_TRIALS = RestlessMultiArmedBandit.N_TRIALS

class QLearningAgent:
    """
    Q-Learning agent for the restless multi-armed bandit problem.
    
    Attributes:
        learning_rate (float): The rate at which the agent updates its Q-values.
        exploration_rate (float): The probability of choosing a random action (exploration) versus the best known action (exploitation).
        q_table_init (torch.Tensor): The initial Q-values for each action.
        q_table (torch.Tensor): The current Q-values for each action.
    """
    def __init__(self, learning_rate, exploration_rate, q_table_init) -> None:
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table_init = torch.tensor(q_table_init, dtype=torch.float32, device=device)
        self.q_table = self.q_table_init.clone()
    
    def reset_q_table(self) -> None:
        """
        Resets the Q-table to its initial values.
        """
        self.q_table = self.q_table_init.clone()
    
    def choose(self) -> int:
        """
        Chooses an action based on the exploration-exploitation trade-off.
        
        Returns:
            int: The chosen action.
        """
        # Implement an exploration mechanism using the exploration rate, so that your agent will mostly exploit the knowledge it has gained so far and pick the straightforward choice, the one which it values the most.
        # However every now and then, depending on the exploration rate, your agent will pick randomly.
        
        ##### INSERT YOUR CODE HERE! #####

        return None # delete this line after implementing the function.
    
    def update(self, action: int, reward: float) -> None:
        """
        Updates the Q-value for the given action based on the received reward.
        
        Args:
            action (int): The action taken.
            reward (float): The reward received for taking the action.
        """
        # Implement the Q-learning update rule, so that your agent can learn from the rewards it receives.

         ##### INSERT YOUR CODE HERE! #####

        return None # delete this line after implementing the function.

def run_episdoe(agent, env):
    """
    Runs a single episode of the environment with the given agent.
    
    Args:
        agent (QLearningAgent): The Q-learning agent.
        env (RestlessMultiArmedBandit): The environment.
        
    Returns:
        Tuple: Containing rewards, actions, Q-table updates, and bandit reward functions for the episode.
    """
    # Initialize the arrays to store the results.
    episode_rewards = np.zeros(N_TRIALS)
    episode_actions = np.zeros(N_TRIALS)
    episode_q_table = np.zeros((N_TRIALS, N_ARMS))
    # Reset the environment and the agent q-table.
    state = env.reset()
    # reset the agent's Q-table.
    agent.reset_q_table()
    # store the initial Q-table.
    episode_q_table[0] = agent.q_table.cpu().detach().numpy()
    # run the episode.


    ##### INSERT YOUR CODE HERE! #####



    return None # delete this line after implementing the function.
  
def test_QlearningAgent(agent, env, num_episodes : int):
    """
    Tests the Q-learning agent over multiple episodes and collects statistics.
    
    Args:
        agent (QLearningAgent): The Q-learning agent.
        env (RestlessMultiArmedBandit): The environment.
        num_episodes (int): The number of episodes to run.
        
    Returns:
        Tuple: Containing rewards, actions, bandits, and Q-table updates for all episodes.
    """
    # Initialize the arrays to store the results.
    all_episodes_rewards = np.zeros((num_episodes, N_TRIALS))
    all_episodes_actions = np.zeros((num_episodes, N_TRIALS))
    all_episodes_q_tables = np.zeros((num_episodes, N_TRIALS, N_ARMS))
    all_episodes_bandits = []

    # Run the episodes and save the results.
    for episode in range(num_episodes):
        episode_rewards, episode_actions, episode_q_table, episode_bandit = run_episdoe(agent, env)
        all_episodes_rewards[episode] = episode_rewards
        all_episodes_actions[episode] = episode_actions
        all_episodes_q_tables[episode] = episode_q_table
        all_episodes_bandits.append(episode_bandit)
    return all_episodes_rewards, all_episodes_actions, all_episodes_bandits, all_episodes_q_tables