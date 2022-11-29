from typing import (
    Tuple,
)

import torch

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    BatchIndices,
    BatchImportance,
    TensorN1,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            full_sink: bool = True,
    ) -> None:
        self.__device = device          # torch.device("cpu") or torch.device("cuda")
        self.__capacity = capacity      # The capacity defines the number of [channels, 84, 84] states that can be stored
        self.__size = 0                 # The size is the number of states that are currently stored, currently empty
        self.__pos = 0                  # The position is the index of the next state to be stored, currently 0
        self.__channels = channels      # The number of channels of the state

        sink = lambda x: x.to(device) if full_sink else x   # Move tensor to device if set fully sink
        self.__m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))   # A capacity lengthed array of [channels, 84, 84] shaped states
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))   # A capacity lengthed array-like object for storing the actions
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))   # A capacity lengthed array-like object for storing the rewards
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))     # A capacity lengthed array-like object for storing the dones

        # Create priorities for the memory
        self.__m_priorities = sink(torch.zeros((capacity, 1), dtype=torch.float32))  # A capacity lengthed array-like object for storing the priorities

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        # Push a new state into the memory
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        # Add the priority of the new state to the memory with the maximum priority and 1 if the memory is empty
        self.__m_priorities[self.__pos, 0] = self.__m_priorities.max() if self.__size > 0 else 1.0

        self.__pos += 1                                 # Increment the position for next insertion
        self.__size = max(self.__size, self.__pos)      # If the memory is not full, the size will increase, otherwise it will stay the same
        self.__pos %= self.__capacity                   # If the memory is full, the position will be reset to 0, deleting the oldest state next time

    def get_probabilities(self, priority_scale: float) -> TensorN1:
        # Get the probabilities of the states in the memory
        scaled_priorities = self.__m_priorities ** priority_scale
        sample_probabilities = scaled_priorities / scaled_priorities.sum()
        return sample_probabilities

    def get_importance(self, probabilities: TensorN1) -> TensorN1:
        # The probabilities is a tensor with shape (N, 1)
        # Get the normalized importance of the probabilities
        importance = (1.0 / len(self.__size)) * (1.0 / probabilities)
        importance_normalized = importance / importance.max()
        return importance_normalized

    def sample(self, batch_size: int, priority_scale: float = 1.0) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
            BatchIndices,
            BatchImportance
    ]:
        # Sample a batch of states from the memory based on the priorities
        probabilities = self.get_probabilities(priority_scale)
        b_indices = torch.multinomial(probabilities, batch_size, replacement=False)
        # indices = torch.randint(0, high=self.__size, size=(batch_size,))    # The indices of the states to be sampled
        b_state = self.__m_states[b_indices, :4].to(self.__device).float()
        b_next = self.__m_states[b_indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[b_indices].to(self.__device)
        b_reward = self.__m_rewards[b_indices].to(self.__device).float()
        b_done = self.__m_dones[b_indices].to(self.__device).float()
        b_importance = self.get_importance(probabilities[b_indices]).to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done, b_indices, b_importance
    
    def update_priorities(self, indices: TensorN1, errors: TensorN1, offset: float = 0.1) -> None:
        # For each pair of indices and errors, update the priorities of the states in the memory
        for index, error in zip(indices, errors):
            self.__m_priorities[index] = abs(error) + offset
    def __len__(self) -> int:
        return self.__size
