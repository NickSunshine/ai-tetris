import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):  # Reduced features_dim
        super().__init__(observation_space, features_dim)
        # Define a CNN with downsampling
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 18, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to (16, 9, 5)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 9, 5)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to (32, 4, 2)
            nn.Flatten()  # Flatten the output for the fully connected layer
        )
        # Compute the size of the flattened output
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        # Define a fully connected layer to produce the desired feature dimension
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))