class GameStateContext: 
    """
    a blackboard class for the Behaviour Tree Agent to read the current game info 
    """ 
    def __init__(self) -> None:
        self.player_x = 0
        self.player_y = 0
        self.enemy_x = 0
        self.enemy_y = 0

    @property
    def get_distance_x(self) -> float:
        return abs(self.player_x - self.enemy_x)
    