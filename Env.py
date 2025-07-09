import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from typing import List, Tuple
from utils import device

class RestlessMultiArmedBandit:
    N_TRIALS = 200  # Define the number of trials
    N_ARMS = 3  # Define the number of arms
    """This class represents an enviroment of RMAB task. The enviroment is defined by a single or multiple sets of bandit reward functions. Each set is a matrix of shape (N_ARMS, N_TRIALS) where N_ARMS is the number of arms and N_TRIALS is the number of trials. 
    Each time you reset the enviroment, a specific set of bandit reward functions is chosen randomly from the list of bandit reward functions sets.
    You interacts with this enviroment by taking a step with a specific action and reciving the next state, based on the reward you got and update trial number you are in."""

    def __init__(self, bandit_rew_list: List[np.ndarray]):
        """
        Initializes the RestlessMultiArmedBandit environment.

        Args:
            bandit_rew_list (List[np.ndarray]): A list of arrays containing rewards for each episode.
        """
        self.bandit_rew_list = bandit_rew_list
        self.reset()

    def reset(self) -> torch.Tensor:
        """
        Resets the environment to an initial state with a random bandit reward functions. 

        Returns:
            torch.Tensor: An initial state.
        """
        # Reset trial number
        self.trial = 0
        # Choose a random bandit reward functions from the list of bandit reward functions 
        self.bandit_rew = self.bandit_rew_list[np.random.choice(len(self.bandit_rew_list))]
        # Initialize the bandit with a random permutation of the arms
        self.bandit = np.random.permutation(self.bandit_rew)
        
        # Create the new state tensor directly
        initial_state = torch.cat([
            torch.tensor([self.trial], dtype=torch.float32, device=device), # trial 0
            torch.tensor([0], dtype=torch.float32, device=device), # reward 0 
            F.one_hot(torch.tensor(np.random.choice([0,1])), num_classes= RestlessMultiArmedBandit.N_ARMS).float().to(device) #one-hot random action
        ])

        return initial_state

    def step(self, action: int) -> torch.Tensor:
        """
        Get the reward according to the choosen action and create the next state.

        Args:
            action (int): The action chosen by the agent.

        Returns:
            Tuple[torch.Tensor, float]: The next state and the reward obtained.
        """
        # Get reward based on the bandit's arm reward function 
        reward = self.bandit[action, self.trial] 

        # Create the new state tensor directly
        next_state = torch.cat([
            torch.tensor([self.trial], dtype=torch.float32, device=device), # trial
            torch.tensor([reward], dtype=torch.float32, device=device), # reward
            F.one_hot(torch.tensor(action), num_classes= RestlessMultiArmedBandit.N_ARMS).float().to(device) #one-hot previous action
        ])
        # update the trial number
        self.trial += 1
        return next_state
    
    def get_episode_bandit(self) -> List[float]:
        """
        Get the current episode's bandit value functions.
        Returns:
            List[float]: The current episode's bandit value functions
        """
        return self.bandit
    
    
    @staticmethod
    def score(episode_rewards: List[int], episode_bandit : np.ndarray) -> int:
        """
        Get the rate between total reward obtained in the episode to the maximum reward obtainable in the episode.

        Args:
            episode_rewards (List[int]): The rewards obtained in the episode.
            episode_bandit (np.ndarray): The bandit reward functions for the episode.

        Returns:
            int: The rate between total reward obtained in the episode to the maximum reward obtainable in the episode.
        """
        return round(sum(episode_rewards) / np.sum(episode_bandit.max(axis=0)), 2)
    
    @staticmethod
    def mean_score(episodes_rewards, episodes_bandits, verbose = True):
        mean_score = round(np.mean([RestlessMultiArmedBandit.score(ep_rewards, ep_bandit) for ep_rewards, ep_bandit in zip(episodes_rewards, episodes_bandits)]),2)
        if verbose:
            print(f'Mean score: {mean_score:.2f}')
        return mean_score
    
def plot_restless_episode(rewards, actions, bandit):
    """
    Plot the reward functions and actions over the course of an episode.

    Args:
        episode_rewards (np.ndarray): Array of rewards obtained in the episode.
        episode_actions (np.ndarray): Array of actions taken in the episode.
        episode_bandit (np.ndarray): Array of bandit reward functions for the episode.
        score (float): The score of the episode.
    """
    num_trials = RestlessMultiArmedBandit.N_TRIALS
    num_arms = bandit.shape[0]
    plt.figure(figsize=(15, 6))
    # Plot reward functions
    for arm in range(num_arms):
        plt.plot(range(num_trials), bandit[arm], label=f'Reward function of arm {arm}')   
    # Scatter plot of actions
    max_reward_actions = np.argmax(bandit, axis=0)
    correct_actions = (actions == max_reward_actions)
    for t in range(num_trials):
        marker = 'o' if correct_actions[t] else 'x'
        plt.scatter(t, rewards[t], color='black', marker=marker)
    # Adding empty scatter plots for legend entries
    plt.scatter([],[], marker='o', color='black', label='Optimal Action')
    plt.scatter([],[],  marker='x', color='black', label='Suboptimal Action')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.legend(loc='best')
    plt.title(f'Reward Functions over Time - Episode Score: {RestlessMultiArmedBandit.score(rewards,bandit):.2f}')
    plt.tight_layout()
    plt.show()