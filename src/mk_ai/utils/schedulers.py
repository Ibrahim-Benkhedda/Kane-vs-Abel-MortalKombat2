import math
from typing import Callable

class Schedules:
    """
    Schedules for RL training.
    """
    @staticmethod
    def exponential_decay(start_rate: float, decay_factor: float) -> Callable[[float], float]:
        """
        Exponential decay schedule.
        
        Parameters:
            start_rate (float): The initial learning rate.
            decay_factor (float): The decay factor.
        
        Returns:
            Callable[[float], float]: A function that takes a progress value and returns the learning rate.
        """
        def schedule(progress: float) -> float:
            return start_rate * (decay_factor ** (1.0 - progress))
        return schedule

    @staticmethod
    def linear_decay(start_rate: float, end_rate: float) -> Callable[[float], float]:
        """
        Linear decay schedule.
        
        Parameters:
            start_rate (float): The initial learning rate.
            end_rate (float): The final learning rate.
        
        Returns:
            Callable[[float], float]: A function that takes a progress value and returns the learning rate.
        """
        def schedule(progress: float) -> float:
            return end_rate + (start_rate - end_rate) * progress
        return schedule
    
    @staticmethod
    def cyclical_lr(start_rate: float, max_rate: float, step_size: float) -> Callable[[float], float]:
        """
        Cyclical learning rate schedule.
        
        Parameters:
            start_rate (float): The initial learning rate.
            max_rate (float): The maximum learning rate.
            step_size (float): The step size for the cyclical schedule.
        
        Returns:
            Callable[[float], float]: A function that takes a progress value and returns the learning rate.
        """
        def schedule(progress: float) -> float:
            cycle = math.floor(1 - progress) / (2 * step_size)
            x = abs((1 - progress) / step_size - 2 * cycle + 1)
            return start_rate + (max_rate - start_rate) * max(0, 1 - x)
        return schedule

