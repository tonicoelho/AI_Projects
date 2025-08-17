import time
import random
import numpy as np
import torch
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from pac_trainer import *

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNets trained in the pac_trainer.py
    module.
    '''

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, really just needs the model and
        its plan Queue.
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained. (Will be
        same format as Constants.MAZE)
        """
        # Task 7
        self.model = PacNet(maze)
        self.model.load_state_dict(torch.load(Constants.PARAM_PATH))
        self.model.eval()

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: String action choice from the set of legal_actions
        """
        # Task 8
        epsilon = 0.1  # Probability of choosing a random action (exploration)
        with torch.no_grad():
            x = PacmanMazeDataset.vectorize_maze(perception).to(Constants.DEVICE)
            outputs = self.model(x)
            outputs = outputs.cpu().numpy()

            if random.random() < epsilon:
                # Choose a random action (exploration)
                best_move = random.choice(legal_actions)[0]
                print("Chosen Move: Random - ", best_move)  # Debug
            else:
                # Choose the action with the highest score (exploitation)
                best_score = -float('inf')
                best_move = None
                for move in legal_actions:
                    idx = PacmanMazeDataset.move_indexes[move[0]]
                    if outputs[idx] > best_score:
                        best_score = outputs[idx]
                        best_move = move[0]
                print("Chosen Move: Best - ", best_move)  # Debug

            return best_move