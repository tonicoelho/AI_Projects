'''
Contains a variety of tests to validate the inner-workings of your
Pacman Trainer system
'''
import pandas as pd
import math
import itertools
import unittest
import pytest
import torch
from pac_trainer import *
from pacman_agent import *
from maze_problem import *
from environment import *

class PacTrainerTests(unittest.TestCase):
    '''
    Simple checkpoint unit tests for the Pacman Trainer and Agent
    
    [!] IMPORTANT: Many tests will only work for the default maze in Constants.MAZE
    MAZE = ["XXXXXXXXX",
            "X..O...PX",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X.......X",
            "XXXXXXXXX"]
    '''
    
    # Dataset Tests
    # -------------------------------------------------------------------------------------------------
    def test_dataset_maze_vectorizer(self) -> None:
        maze = [
            "XXXX",
            "XO.X",
            "X.PX",
            "XXXX"
        ]
        
        vectorized_maze = PacmanMazeDataset.vectorize_maze(maze)
        self.assertEqual(torch.float, vectorized_maze.dtype, "[X] Your vectorized maze is not the right data type, make sure to convert it to torch.float")
        
        answer = torch.tensor([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=torch.float)
        self.assertTrue(torch.equal(answer, vectorized_maze),
                        "[X] Your vectorize_maze is not converting maze entities in the correct format;\
                        see maze_entity_indexes in the PacmanMazeDataset class. Your maze:\n" + str(vectorized_maze))
    
    def test_dataset_move_vectorizer(self) -> None:
        vectorized_move = PacmanMazeDataset.vectorize_move("R")
        self.assertEqual(torch.float, vectorized_move.dtype, "[X] Your vectorized move is not the right data type, make sure to convert it to torch.float")
        MOVE_ERR = "[X] Your vectorized move does not match the expected output: a 1D, one-hot tensor of floats with indexes corresponding to PacmanMazeDataset.move_indexes"
        self.assertTrue(torch.equal(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float), vectorized_move), MOVE_ERR)
        vectorized_move = PacmanMazeDataset.vectorize_move("U")
        self.assertTrue(torch.equal(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float), vectorized_move), MOVE_ERR)
    
    def test_pacnet_outputs(self) -> None:
        pacnet = PacNet(Constants.MAZE).to(Constants.DEVICE)
        vectorized_maze = PacmanMazeDataset.vectorize_maze(Constants.MAZE)
        try:
            outputs = pacnet(vectorized_maze)
        except:
            pytest.fail("[X] Assuming your vectorized maze is correct, your PacNet may not be configured correctly to take a vector of its size in the input layer (other issues possible)")
        self.assertEqual(torch.float, outputs.dtype, "[X] Your move activation outputs is not the right data type, make sure to convert it to torch.float")
        self.assertEqual(torch.Size([4]), outputs.size(), "[X] Your outputs should be a 1D tensor of floats with precisely 4 values")
        
    def test_pacman_agent_choice(self) -> None:
        try:
            agent = PacmanAgent(Constants.MAZE)
        except:
            pytest.fail("[X] Will only function properly for a PacNet that has already been trained and has its parameters saved via pac_trainer")
        mp = MazeProblem(Constants.MAZE)
        choice = agent.choose_action(Constants.MAZE, mp.legal_actions(mp.get_player_loc()))
        self.assertIn(choice, Constants.MOVES, "[X] Your agent returned something that wasn't one of the valid action choices: ['U', 'D', 'L', 'R']")
        
    def test_integration(self) -> None:
        env = Environment(Constants.MAZE, window=None, tick_length=0, verbose=False, debug=False)
        while env.get_score() > Constants.get_min_score() and env.get_pellets_eaten() < 4:
            env.move()
        self.assertEqual(4, env.get_pellets_eaten(), "[X] Your agent failed to eat all of the pellets before the game ended")
        self.assertLess(-20, env.get_pellets_eaten(), "[X] Your agent did not eat the pellets using a path that was near enough to optimal")
        
if __name__ == '__main__':
    unittest.main()
