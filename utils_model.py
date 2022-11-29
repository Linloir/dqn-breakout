import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    
    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        # The input is a stack of 4 frames, with shape (1, 4, 84, 84), which has 4 channels and 84x84 pixels for each channel
        # Define the Deep Q Network with 3 convolutional layers and 2 fully connected layers, which generates an action vector out of the input
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)    # Convolutional layer with a 4 channel input, 32 channel output, 8x8 kernel size, 4 stride
        # After the first convolutional layer, the output has 32 channels and 20x20 pixels for each channel
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)   # Convolutional layer with a 32 channel input, 64 channel output, 4x4 kernel size, 2 stride
        # After the second convolutional layer, the output has 64 channels and 9x9 pixels for each channel
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)   # Convolutional layer with a 64 channel input, 64 channel output, 3x3 kernel size, 1 stride
        # After the third convolutional layer, the output has 64 channels and 7x7 pixels for each channel
        self.__fc1 = nn.Linear(64*7*7, 512)                          # Fully connected layer with a [64, 7, 7] input, 512 output
        self.__fc2 = nn.Linear(512, action_dim)                      # Fully connected layer with a 512 input and the output size of the action space
        self.__device = device

    def forward(self, x):
        x = x / 255.                                            # Normalize the input to [0, 1]
        x = F.relu(self.__conv1(x))                             # Apply the first convolutional layer and ReLU activation
        x = F.relu(self.__conv2(x))                             # Apply the second convolutional layer and ReLU activation
        x = F.relu(self.__conv3(x))                             # Apply the third convolutional layer and ReLU activation
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))           # Apply the first fully connected layer and ReLU activation
        return self.__fc2(x)                                    # Apply the second fully connected layer

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
