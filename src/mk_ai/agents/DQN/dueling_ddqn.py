from .double_dqn import DoubleDQN
from .dueling_dqn import DuelingCnnPolicy

# -------------------------
# 5) DuelingDoubleDQN
# -------------------------
class DuelingDoubleDQN(DoubleDQN):
    """
    DuelingDoubleDQN combines the Double DQN update logic with a dueling network architecture.
    it inherits the training and update logic from DoubleDQN and automatically sets a dueling CNN-based
    policy as the default. By default, if no feature extractor is provided, it uses NatureCNN.
    
    The dueling architecture splits the Q-network into two streams:
    
      - A value stream that estimates the state-value V(s)
      - An advantage stream that estimates the advantages A(s,a)
    
    These streams are then combined to obtain the final Q-value estimates as:
    
      Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    
    Parameters:
        *args: Positional arguments passed to the parent DoubleDQN class.
        **kwargs: Keyword arguments passed to the parent DoubleDQN class.
                  If no 'policy' is provided, this defaults to DuelingCnnPolicy.
                  Additionally, if no 'features_extractor_class' is provided in policy_kwargs,
                  DuelingCnnPolicy will force it to NatureCNN.
    """

    def __init__(self, *args, **kwargs):
        # If no policy is specified, default to DuelingCnnPolicy
        if "policy" not in kwargs:
            kwargs["policy"] = DuelingCnnPolicy
        super().__init__(*args, **kwargs)
