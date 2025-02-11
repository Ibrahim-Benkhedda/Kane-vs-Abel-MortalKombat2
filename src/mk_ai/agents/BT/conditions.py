def is_enemy_to_the_right(ctx) -> bool:
    """
    Returns True if the player_x is significantly left of enemy_x,
    meaning the enemy is to the right.
    """
    return ctx.player_x < (ctx.enemy_x - 50)

def is_enemy_to_the_left(ctx) -> bool:
    """
    Returns True if the player_x is significantly right of enemy_x,
    meaning the enemy is to the left.
    """
    return ctx.player_x > (ctx.enemy_x + 50)

def is_close_to_enemy(ctx) -> bool:
    """
    Returns True if the player and enemy are within 50 units horizontally.
    """
    return abs(ctx.player_x - ctx.enemy_x) <= 50
