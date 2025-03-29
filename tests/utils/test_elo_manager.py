import unittest
from src.mk_ai.utils.elo_manager import EloManager
import os
import json

class TestEloManager(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_elo.json"
        self.elo_manager = EloManager(self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_init_default_values(self):
        self.assertEqual(self.elo_manager.default_rating, 1500)
        self.assertEqual(self.elo_manager.k_factor, 32)
        self.assertEqual(self.elo_manager.file_path, self.test_file)

    def test_get_rating_new_agent(self):
        rating = self.elo_manager.get_rating("new_agent")
        self.assertEqual(rating, 1500)

    def test_update_ratings_win(self):
        self.elo_manager.update_ratings("player1", "player2", "player1")
        self.assertGreater(self.elo_manager.get_rating("player1"), 1500)
        self.assertLess(self.elo_manager.get_rating("player2"), 1500)

    def test_update_ratings_persistence(self):
        self.elo_manager.update_ratings("player1", "player2", "player1")
        original_ratings = self.elo_manager.ratings.copy()
        
        # Create new manager instance to test file loading
        new_manager = EloManager(self.test_file)
        self.assertEqual(original_ratings, new_manager.ratings)

    def test_invalid_file_path(self):
        invalid_manager = EloManager("/invalid/path/ratings.json")
        self.assertEqual(invalid_manager.ratings, {})

    def test_corrupted_file(self):
        # Create corrupted JSON file
        with open(self.test_file, 'w') as f:
            f.write("corrupted json{")
        
        manager = EloManager(self.test_file)
        self.assertEqual(manager.ratings, {})

if __name__ == '__main__':
    unittest.main()