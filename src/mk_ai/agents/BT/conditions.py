from mk_ai.agents.BT.game_context import GameStateContext
from typing import Dict, Callable, Any

class ConditionsProvider:
    """
    Provides behavior tree conditions and generates condition mappings.
    """
    @staticmethod
    def is_enemy_to_the_right(ctx: GameStateContext) -> bool:
        """
        Returns True if the player_x is significantly left of enemy_x,
        meaning the enemy is to the right.
        """
        return ctx.player_x < (ctx.enemy_x - 50)
    
    @staticmethod
    def is_enemy_to_the_left(ctx: GameStateContext) -> bool:
        """
        Returns True if the player_x is significantly right of enemy_x,
        meaning the enemy is to the left.
        """
        return ctx.player_x > (ctx.enemy_x + 50)
    
    @staticmethod
    def is_close_to_enemy(ctx: GameStateContext) -> bool:
        """
        Returns True if the player and enemy are within 50 units horizontally.
        """
        return abs(ctx.player_x - ctx.enemy_x) <= 50
    
    @staticmethod
    def is_long_range_enemy(ctx: GameStateContext) -> bool:
        """
        Returns True if the enemy is further than 50 units away horizontally.
        """
        return abs(ctx.player_x - ctx.enemy_x) > 50
    
    @staticmethod
    def is_medium_range_enemy(ctx: GameStateContext) -> bool:
        """
        Determines if enemy is at a medium range - not too close, not too far.
        Good for jump-in attacks.
        """
        distance = abs(ctx.player_x - ctx.enemy_x)
        # Adjust these values based on your game's specific spacing
        return 50 <= distance <= 200
     
    @classmethod
    def gen_condition_map(cls) -> Dict[str, Callable[[Any], bool]]:
        """
        Generates a dictionary mapping condition names to condition functions.
        """
        condition_map = {}
        for name in dir(cls):
            if name.startswith('is_') and callable(getattr(cls, name)):
                condition_map[name] = getattr(cls, name)
                
        return condition_map


# This is For backward compatibility which maintains the original function references
is_enemy_to_the_right = ConditionsProvider.is_enemy_to_the_right
is_enemy_to_the_left = ConditionsProvider.is_enemy_to_the_left
is_close_to_enemy = ConditionsProvider.is_close_to_enemy
is_long_range_enemy = ConditionsProvider.is_long_range_enemy
is_medium_range_enemy = ConditionsProvider.is_medium_range_enemy