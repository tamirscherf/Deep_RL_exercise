import torch
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
import numpy as np
from Env import RestlessMultiArmedBandit
from ActorCriticAgent import ActorCriticAgent
from utils import device, RUNS_DIR, double_unsqz 
from torch.utils.tensorboard import SummaryWriter

N_ARMS = RestlessMultiArmedBandit.N_ARMS
N_TRIALS = RestlessMultiArmedBandit.N_TRIALS

def compute_returns(rewards: Tensor, gamma: float) -> Tensor:
    """
    Returns a tensor of the cumulative discounted rewards at each time step.

    Args:
        rewards (Tensor): The rewards received at each time step.
        gamma (float): The discount factor.

    Returns:
        Tensor: The cumulative discounted rewards.
    """
    
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in reversed(range(rewards.shape[0] - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns

def compute_loss(actions_log_probs: Tensor, values: Tensor, returns: Tensor, actions: Tensor, beta_critic: float, beta_entropy: float) -> Tensor:
    """
    Computes the combined actor-critic loss.

    Args:
        actions_log_probs (Tensor): Log probabilities of actions taken.
        values (Tensor): Values of the states.
        returns (Tensor): Cumulative discounted rewards.
        actions (Tensor): Actions taken.
        beta_critic (float): Weight of the critic loss.
        beta_entropy (float): Weight of the entropy term.

    Returns:
        Tensor: The combined loss.
    """
    
    ##### INSERT YOUR CODE HERE! #####

    return None # delete this line after implementing the function.

def compute_accuracy(rewards: Tensor, episode_bandit: np.ndarray) -> float:
    """
    Computes the accuracy of the agent based on the rewards received.

    Args:
        rewards (Tensor): Rewards received.
        episode_bandit (np.ndarray): The bandit configuration for the episode.

    Returns:
        float: The accuracy of the agent.
    """
    return torch.sum(rewards) / np.sum(episode_bandit.max(axis=0))

def run_episode(initial_state: Tensor, hidden_states, model: ActorCriticAgent, env: RestlessMultiArmedBandit):
    """
    Run the model for one episode.

    Args:
        initial_state (Tensor): The initial state tensor.
        hidden_states (tuple): The initial hidden states for the LSTM.
        model (ActorCriticAgent): The actor-critic model.
        env (RestlessMultiArmedBandit): The environment.

    Returns:
        Tuple: Log probabilities, values, rewards, actions, and hidden states for the episode.
    """
    state = initial_state
    values = torch.zeros(N_TRIALS, dtype=torch.float32, device=device)
    actions = torch.zeros(N_TRIALS, dtype=torch.float32, device=device)
    rewards = torch.zeros(N_TRIALS, dtype=torch.float32, device=device)
    actions_log_probs = torch.zeros(N_TRIALS, N_ARMS, dtype=torch.float32, device=device)

    
    # run the episode


    ##### INSERT YOUR CODE HERE! #####


    return None # delete this line after implementing the function.

def train(model: ActorCriticAgent, num_episodes: int, env: RestlessMultiArmedBandit, print_every: int, cfg, write=False, name='NO_NAME'):
    """
    Trains the model over multiple episodes.

    Args:
        model (ActorCriticAgent): The actor-critic model.
        num_episodes (int): Number of episodes to train.
        env (RestlessMultiArmedBandit): The environment.
        print_every (int): Frequency of printing the training progress.
        cfg (dict): Configuration dictionary with training parameters.
        write (bool): Whether to write logs to TensorBoard.
        name (str): Name for the TensorBoard log directory.

    Returns:
        Tuple: The trained model, final hidden states, and optimizer.
    """
    # Initialize hidden states
    hidden_states = (torch.zeros(1, 1, cfg['hidden_size'], dtype=torch.float32, device=device), 
                     torch.zeros(1, 1, cfg['hidden_size'], dtype=torch.float32, device=device))

    if write:
        writer = SummaryWriter(log_dir=(RUNS_DIR / name))

    # Set the model to training mode and initialize the optimizer and scheduler

    ##### INSERT YOUR CODE HERE! #####

    for epsd in tqdm(range(num_episodes)):
        # Reset environment
        ##### INSERT YOUR CODE HERE! #####
        # Run an episode
        #### INSERT YOUR CODE HERE! #####
        # Calculate loss
        #### INSERT YOUR CODE HERE! #####
        # Optimize, use retain_graph=True with the backward pass
        #### INSERT YOUR CODE HERE! #####

        # Update the scheduler
        #### INSERT YOUR CODE HERE! #####

        # Evaluate and log
        if write:
            rew_perc = compute_accuracy(rewards, env.get_episode_bandit())
            writer.add_scalar("Train/Accuracy", rew_perc, epsd)
            writer.add_scalar("Loss/final_loss", final_loss, epsd)
            writer.add_scalar("Loss/critic_loss", critic_loss, epsd)
            writer.add_scalar("Loss/actor_loss", actor_loss, epsd)
            writer.add_scalar("Loss/entropy_loss", entropy, epsd)
        # Print the progress
        if epsd % print_every == 0:
            if not write:  # If write is False we should calculate the reward percentage
                rew_perc = compute_accuracy(rewards, env.get_episode_bandit())
            print(f'Episode {epsd}  Accuracy: {rew_perc:.2f}  Loss: {final_loss:.2f}')
            
    return model, hidden_states, optimizer



