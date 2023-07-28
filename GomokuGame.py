
from gomoku import Gomoku  # assuming gomoku.py is the file containing the Gomoku class
from Player import Player, HumanPlayer  # assuming player.py is the file containing the Player classes
import logging
# Create a logger
logger = logging.getLogger('gomoku_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('gomoku_games.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

class GomokuGame:
    def __init__(self, player1, player2):
        self.game = Gomoku()
        # Map 'X' and 'O' to player1 and player2
        self.players = {'X': player1, 'O': player2}

    def start_game(self):
        print("Welcome to the Gomoku game!")
        logger.info('New Game')
        self.game.print_board()
        while not self.game.get_game_state()['game_over']:
            print(f"Player {self.game.current_player}'s turn.")
            
            # Get the move from the current player
            x, y = self.players[self.game.current_player].get_move(self.game.get_game_state())

            # Make the move
            try:
                self.game.move(x, y)
            except Exception as e:
                print(e)
                continue

            logger.info(f'{self.game.current_player}: {x},{y}')

            # Print the updated board
            self.game.print_board()

            # Check if the game has ended
            if self.game.get_game_state()['game_over']:
                print(f"Player {self.game.get_game_state()['winner']} has won!")
                logger.info(f"Player {self.game.get_game_state()['winner']} has won!")
                break

        print("Thank you for playing!")

# To play the game, create two human players, an instance of GomokuGame with the players, and call the start_game method:
player1 = HumanPlayer()
player2 = HumanPlayer()
game = GomokuGame(player1, player2)
game.start_game()
