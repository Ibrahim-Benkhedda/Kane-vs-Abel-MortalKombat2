import json
import os
from typing import Dict

class EloManager:
    """
    Manages Elo ratings for agents in a game. Ratings are stored in a JSON file.
    """  
    def __init__(self, file_path: str = "elo_ratings.json", default_rating: int = 1500, k_factor: int = 32):
        self.file_path = file_path
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = self.load_ratings() # Load ratings from file

    def load_ratings(self) -> Dict[str, float]:
        """Load the Elo ratings from a file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading Elo ratings: {e}")
        return {}

    def save_ratings(self) -> None:
        """Save the Elo ratings to a file"""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.ratings, f, indent=4)
        except Exception as e:
            print(f"Error saving Elo ratings: {e}")

    def get_rating(self, agent_id: str) -> float:
        """Get the Elo rating for the specified agent"""
        return self.ratings.get(agent_id, self.default_rating)

    def update_ratings(self, agent_a: str, agent_b: str, winner: str) -> None:
        """
        Update ratings based on the match result.

        Parameters
            agent_a: identifier for player 1
            agent_b: identifier for player 2
            winner: identifier of the winning agent (must be either agent_a or agent_b)
        """
        # Get current ratings
        rating_a = self.get_rating(agent_a)
        rating_b = self.get_rating(agent_b)

        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

        # Calculate new ratings
        score_a = 1 if winner == agent_a else 0
        score_b = 1 if winner == agent_b else 0

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        # Save new ratings
        self.ratings[agent_a] = new_rating_a
        self.ratings[agent_b] = new_rating_b

        # Save ratings to file
        self.save_ratings()

        # Print updated ratings
        print(f"Updated Elo ratings: {agent_a} => {new_rating_a:.1f}, {agent_b} => {new_rating_b:.1f}")


