from stable_baselines3.common.callbacks import EvalCallback

class CustomEvalCallback(EvalCallback):
    """
    Custom EvalCallback that evaluates the model at the start of training
    and at specified intervals.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set last_eval to -eval_freq to trigger evaluation immediately at step 0
        self._last_eval = -self.eval_freq
