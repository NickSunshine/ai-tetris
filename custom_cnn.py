import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        # Define a CNN with downsampling
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 9, 5)
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Batch normalization
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 9, 5)
            nn.ReLU(),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to (64, 4, 2)
            nn.Flatten()  # Flatten the output for the fully connected layer
        )
        # Compute the size of the flattened output
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        # Define fully connected layers to produce the desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),  # Additional fully connected layer
            nn.ReLU(),
            nn.Linear(256, features_dim),  # Map to the desired feature dimension
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))