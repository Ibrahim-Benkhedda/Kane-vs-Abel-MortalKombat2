import retro

# List available games
games = retro.data.list_games()

# Print the available games
print("Available games:")
for game in games:
    print(game)