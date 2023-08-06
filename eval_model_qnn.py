
from gomoku import Gomoku  # assuming gomoku.py is the file containing the Gomoku class
from GomokuGame import GomokuGame
from Player import Player, HumanPlayer, RandomPlayer, QLearningPlayer, QNNLearningPlayer  # assuming player.py is the file containing the Player classes
from GomokuNet import GomokuLFC1HNNet
import logging
import torch

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


model=GomokuLFC1HNNet(input_size=3*3*3+1, hidden_size=32, output_size=9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# Load the saved weights into the model
try:
    model.load_state_dict(torch.load('GomokuLFC1HNNet_p1.pth'))
except FileNotFoundError as e:
    print(e)

player1 = QNNLearningPlayer(model=model, optimizer=optimizer, board_size=3, training=True, logger=logger)
player1.load_terminal_transactions()
# player1.offline_training(epochs=5, batch_size=1000, max_iter=1000)
player1.eval_model_on_terminal_txn(sample_size=1000)

