import torch
from torch import nn
from utils import init_weights_


class ActorCriticAgent(nn.Module):
    """
    Advantage Actor-Critic (A2C) module combining LSTM, Actor, and Critic modules.

    Attributes:
        lstm (nn.LSTM): The LSTM layer for processing input sequences.
        actor (nn.Linear): The linear layer for generating action probabilities.
        critic (nn.Linear): The linear layer for evaluating state values.
    """
    def __init__(self, input_size: int, hidden_size: int, num_actions: int):
        """
        Initializes the ActorCriticAgent module with LSTM, Actor, and Critic.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The number of hidden units in the LSTM.
            num_actions (int): The number of possible actions.
        """
        super(ActorCriticAgent, self).__init__()
        # Initialize the lstm layer
        #self.lstm = ##### INSERT YOUR CODE HERE! #####
        # Initialize the actor layer
        #self.actor = ##### INSERT YOUR CODE HERE! #####
        # Initialize the critic layer
        #self.critic = ##### INSERT YOUR CODE HERE! #####
        
        init_weights_(self.actor)
        init_weights_(self.critic, gain=0.1)

    def forward(self, state: torch.Tensor, hidden_states):
        """
        Forward pass for the ActorCriticAgent module.

        Args:
            state (torch.Tensor): The input tensor.
            hidden_states: The hidden states of the LSTM.

        Returns:
            tuple: The log probabilities of actions, the value of the state, and the updated hidden states.
        """
        # Implement the forward pass
        
        ##### INSERT YOUR CODE HERE! #####



        return None # delete this line after implementing the function.

    def choose(self, log_probs: torch.Tensor):
        """
        Chooses an action based on the log probabilities of actions.

        Args:
            log_probs (torch.Tensor): The log probabilities of actions.

        Returns:
            int: The chosen action.
        """
        

        
        ##### INSERT YOUR CODE HERE! #####



        return None # delete this line after implementing the function.
