import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN, 
    create_mlp,
)

class DuelingDQN(DQN):
    """
    A DQN that always uses a dueling Q-network by default.
    NOTE: it does NOT override the update rule, so it is *not* Double DQN.
    """

    def __init__(self, *args, **kwargs):
        # If no policy is specified, default to DuelingCnnPolicy
        if "policy" not in kwargs:
            kwargs["policy"] = DuelingCnnPolicy
        super().__init__(*args, **kwargs)

    # Optionally, you can also override other DQN aspects if needed.
    # But by default, this will just build a dueling Q-network.

class DuelingQNetwork(BasePolicy):
    """
    paper: https://arxiv.org/pdf/1511.06581
    Dueling Q-network that separately estimates the state-value V(s)
    and the advantage values A(s,a) from convolutional features,
    and then combines them to obtain the Q-values:
    
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    
    Parameters:
        observation_space (spaces.Space): The observation space of the environment.
        action_space (spaces.Discrete): The discrete action space.
        features_extractor (BaseFeaturesExtractor): The feature extractor module.
        features_dim (int): The number of features output by the feature extractor.
        net_arch (list[int]): The architecture of the fully-connected layers following the CNN features.
        activation_fn (type[nn.Module]): The activation function to use in the MLP layers. Default is nn.ReLU.
        normalize_images (bool): Whether to normalize image inputs (default: True).
    
    Returns:
        Tensor: Q-values with shape [batch_size, action_space.n]
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # Number of actions
        self.action_dim = int(self.action_space.n)
        self.features_dim = features_dim  # Dimension of the extracted CNN features
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        # Build advantage and value streams
        # net_arch controls the fully-connected layers after the CNN features.
        self.advantage_net = nn.Sequential(
            *create_mlp(self.features_dim, self.action_dim, net_arch, activation_fn)
        )
        self.value_net = nn.Sequential(
            *create_mlp(self.features_dim, 1, net_arch, activation_fn)
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Forward pass through the dueling Q-network.

        Parameters:
            obs (PyTorchObs): The observation, a tensor of image data.
        
        Returns:
            th.Tensor: The estimated Q-values for each action.
        """

        # (1) Extract CNN features: shape [batch_size, features_dim]
        features = self.extract_features(obs, self.features_extractor)

        # (2) Compute advantage (A) and value (V)
        advantages = self.advantage_net(features)  # [batch_size, action_dim]
        values = self.value_net(features)          # [batch_size, 1]

        # (3) Combine them: Q = V(s) + A(s,a) - mean(A(s,a'), over a')
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        Q_values = values + (advantages - advantages_mean)
          
        return Q_values

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        """
        Greedy action selection from Q-values.

        Parameters:
            obs (PyTorchObs): The observation, a tensor of image data.
            deterministic (bool): Whether to select the action with the highest Q-value (default: True).
        
        Returns:
            th.Tensor: The selected action indices.
        """
        # Greedy action selection from Q-values
        q_values = self.forward(obs)
        return q_values.argmax(dim=1).reshape(-1)
    
class DuelingCnnPolicy(DQNPolicy):
    """
    A DQN policy that uses a dueling CNN-based Q-network.
    
    This policy uses a dueling architecture after a CNN-based feature extractor.
    If no features extractor is provided in the policy kwargs, it defaults to using NatureCNN.
    
    Args:
        All arguments are passed from DQNPolicy. Optionally, the user may provide
        'features_extractor_class' in the kwargs to override the default.
    """
    def __init__(self, *args, **kwargs):
        # Forces the use of NatureCNN by default if not specified in kwargs.
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = NatureCNN
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> DuelingQNetwork:
        """
        Creates a dueling Q-network for the policy.

        Returns:
            DuelingQNetwork: The dueling Q-network with CNN feature extraction, advantage, and value streams.
        """
        # net_args is populated by DQNPolicy's constructor
        # and includes: observation_space, action_space, net_arch, activation_fn, etc.

        # makes sure a new features_extractor is created each time
        net_args = self._update_features_extractor(
            self.net_args, 
            features_extractor=None
        )

        # Return our custom dueling Q-network
        return DuelingQNetwork(**net_args).to(self.device)
