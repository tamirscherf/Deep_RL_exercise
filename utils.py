import numpy as np
import torch
from torch import nn
import json
from pathlib import Path
import os


PROJECT_DIR = Path(os.path.abspath(os.curdir))
RUNS_DIR = PROJECT_DIR / 'runs'
MODELS_DIR = PROJECT_DIR / 'trained_models/'

device = torch.device("cpu")

def load_reward_functions():
    """
    Load the test reward functions for the restless multi-armed bandit environment.

    Returns:
        List[np.ndarray]: A list of arrays containing rewards for each episode.
    """
    data = np.load('Data/rew_functions.npy')
    return data

def load_training_reward_functions():
    """_summary_
    Load the training reward functions for the restless multi-armed bandit environment.
    Used for training the Actor Critic agent.
    """
    training_data = np.load('Data/training_rew_functions.npz')['array']
    return training_data

def random_argmax(tensor):
    """
    Returns a random index of the maximum value in a given tensor.
    Parameters:
    tensor (torch.Tensor): A 1D tensor of numerical values.
    Returns:
    int: A randomly chosen index of the maximum value in the tensor.
    """
    # Find the maximum value in the tensor
    max_value = tensor.max() 
    # Get all indices where the tensor value equals the maximum value
    max_indices = (tensor == max_value).nonzero(as_tuple=True)[0]
    # Randomly select one of these indices
    random_index = max_indices[torch.randint(len(max_indices), (1,)).item()]
    return random_index

def init_weights_(module: nn.Module, gain: float = 1) -> None:
    """
    Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(module, (nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def double_unsqz(x : torch.Tensor):
    x.unsqueeze_(0)
    x.unsqueeze_(0)
    return x

def load_checkpoint_itmes(fname):
    assert os.path.isfile(fname)
    items = torch.load(MODELS_DIR / fname)
    return items

def save_checkpoint(fname, model,hidden_state, cfg, optimizer=None, verbose = False):
    optimizer_state = optimizer.state_dict() if optimizer else None
    torch.save({'model': model,
                'hidden_state': hidden_state,
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optimizer_state': optimizer_state,
                'cfg': cfg}, MODELS_DIR / fname)
    if verbose:
        print("Saved model to: " + str(fname))