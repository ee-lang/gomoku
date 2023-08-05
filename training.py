
from gomoku import Gomoku  # assuming gomoku.py is the file containing the Gomoku class
from GomokuGame import GomokuGame
from Player import Player, HumanPlayer, RandomPlayer, QLearningPlayer, QNNLearningPlayer  # assuming player.py is the file containing the Player classes
from GomokuNet import GomokuNet
import logging
# Create a logger
logger = logging.getLogger('gomoku_training_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('gomoku_training.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


training_episodes = 10000

# To play the game, create two human players, an instance of GomokuGame with the players, and call the run_game method:
# player1 = HumanPlayer()
# player2 = HumanPlayer()
# player1 = RandomPlayer()
player1 = QLearningPlayer(epsilon=0.001,training=True, logger=logger)

# Load the Q-table from a file
player1.load_q_table('q_table.pkl')

player2 = RandomPlayer()

# # Create a QNNLearningPlayer
# player1 = QNNLearningPlayer(model, optimizer, device='cpu', gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)

# # Create a QNNLearningPlayer
# player2 = QNNLearningPlayer(model, optimizer, device='cpu', gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)

p1_score, p2_score, draws = 0.0, 0.0, 0

for i in range(training_episodes):
    game = GomokuGame(player1, player2, size=3, pieces_in_row_to_win=3, logger=logger)
    result = game.run_game()
    logger.info(f'Episode {i}, {result}')
    p1_score += result[0]
    p2_score += result[1]
    if result[0]==result[1]:
        draws += 1

logger.info(f'Training epoch episodes ({training_episodes}) scores {p1_score}, {p2_score}, draws {draws}')

# Save the Q-table to a file
player1.save_q_table('q_table.pkl')