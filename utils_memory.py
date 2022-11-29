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

        sink = lambda x: x.to(device) if full_sink else x   # Move tensor to device if set fully sink
        self.__m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))   # A capacity lengthed array of [channels, 84, 84] shaped states
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))   # A capacity lengthed array-like object for storing the actions
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))   # A capacity lengthed array-like object for storing the rewards
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))     # A capacity lengthed array-like object for storing the dones

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

        self.__pos += 1                                 # Increment the position for next insertion
        self.__size = max(self.__size, self.__pos)      # If the memory is not full, the size will increase, otherwise it will stay the same
        self.__pos %= self.__capacity                   # If the memory is full, the position will be reset to 0, deleting the oldest state next time

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        # Sample a batch of states from the memory
        indices = torch.randint(0, high=self.__size, size=(batch_size,))    # The indices of the states to be sampled
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size
