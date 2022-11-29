from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device)      # The policy network
        self.__target = DQN(action_dim, device).to(device)      # The target network
        if restore is None:
            # Initialize the weights of the networks
            self.__policy.apply(DQN.init_weights)
        else:
            # Restore the weights of the networks
            self.__policy.load_state_dict(torch.load(restore))
        # Initialize the target network with the same weights as the policy network, making them identical at the beginning
        self.__target.load_state_dict(self.__policy.state_dict())
        # Set the optimizer for the policy network
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False, testing: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if testing or self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""

        # Sample a batch of transitions from the replay memory
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)

        # Double DQN
        # Get the Q(s, a) values from the policy network
        values = self.__policy(state_batch.float()).gather(1, action_batch)
        # Get the action with the highest Q-value from the policy network, which should be argmax_a Q(s', a)
        next_actions = self.__policy(next_batch.float()).max(1).indices.unsqueeze(1)
        # Get the Q(s', argmax_a Q(s', a)) values from the target network
        values_next = self.__target(next_batch.float()).gather(1, next_actions)
        # Calculate the expected Q-value
        expected = reward_batch + self.__gamma * values_next.unsqueeze(1) * (1 - done_batch)
        # Calculate the loss
        loss = F.smooth_l1_loss(values, expected)

        self.__optimizer.zero_grad()                                    # Reset the gradients
        loss.backward()                                                 # Calculate the gradients
        for param in self.__policy.parameters():
            # For each parameter, clip the gradient to [-1, 1]
            param.grad.data.clamp_(-1, 1)
        # Update the weights of the policy network using the gradients
        self.__optimizer.step()

        return loss.item()                                              # Return the loss

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
