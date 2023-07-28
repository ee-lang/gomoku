from gomoku import Gomoku  # assuming gomoku.py is the file containing the Gomoku class


class GomokuGame:
    def __init__(self):
        self.game = Gomoku()

    def start_game(self):
        print("Welcome to the Gomoku game!")
        self.game.print_board()
        # print(self.game.get_game_state())
        while not self.game.get_game_state()['game_over']:
            print(f"Player {self.game.current_player}'s turn.")
            move = input("Enter your move in hexadecimal format (e.g., 'A1'): ")

            # Convert the move to a pair of integers
            x = int(move[0], 16)   # Convert the first part of the move to an integer
            y = int(move[1:], 16)  # Convert the second part of the move to an integer

            # Make the move
            try:
                self.game.move(x, y)
            except Exception as e:
                print(e)
                continue

            # Print the updated board
            self.game.print_board()

            # Check if the game has ended
            if self.game.get_game_state()['game_over']:
                print(f"Player {self.game.get_game_state()['winner']} has won!")
                break

        print("Thank you for playing!")

# To play the game, create an instance of GomokuGame and call the start_game method:
game = GomokuGame()
game.start_game()
