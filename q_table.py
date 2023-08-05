# 2023-08-05 01:25:25,582 - gomoku_training_logger - DEBUG - state (('X', ' ', ' '), (' ', 'X', ' '), ('O', ' ', 'O'))
# 2023-08-05 01:25:25,582 - gomoku_training_logger - DEBUG - available_moves [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1)]


# from gomoku import Gomoku  # assuming gomoku.py is the file containing the Gomoku class
# from GomokuGame import GomokuGame
from Player import Player, HumanPlayer, RandomPlayer, QLearningPlayer, QNNLearningPlayer  # assuming player.py is the file containing the Player classes
# from GomokuNet import GomokuNet
import logging
# Create a logger
logger = logging.getLogger('gomoku_training_logger')
logger.setLevel(logging.DEBUG)

# # Create a file handler
# file_handler = logging.FileHandler('gomoku_training.log')
# file_handler.setLevel(logging.DEBUG)

# # Create a formatter and add it to the file handler
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # Add the file handler to the logger
# logger.addHandler(file_handler)


# training_episodes = 10000

# To play the game, create two human players, an instance of GomokuGame with the players, and call the run_game method:
# player1 = HumanPlayer()
# player2 = HumanPlayer()
# player1 = RandomPlayer()
player1 = QLearningPlayer(epsilon=0.001,training=True, logger=logger)

# Load the Q-table from a file
player1.load_q_table('q_table.pkl')

state = (('X', ' ', ' '), (' ', 'X', ' '), ('O', ' ', 'O'))
available_moves = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1)]
for m in available_moves:
	print(f'({state},{m}) = {player1.q_table[(state,m)]}')