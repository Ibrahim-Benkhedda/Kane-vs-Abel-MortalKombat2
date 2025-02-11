import retro

def print_available_games(search_term=None):
    """
    Print the available games in the Retro environment.
    
    This function was initially used to check whether MortalKombat was available
    during the setup of the environment. it can be used to search for any game by
    specifying a search term.
    
    Args:
        search_term (str, optional): A search term to filter the games,
        defaults to None, which is all games.
    """
    # List available games
    games = retro.data.list_games()
    
    # Filter games if a search term is provided
    if search_term:
        games = [game for game in games if search_term.lower() in game.lower()]
    
    # Print the available games
    if games:
        print("Available games:")
        for game in games:
            print(game)
    else:
        print("No games found matching the search term.")