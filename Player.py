class Player:
    def get_move(self, game_state):
        raise NotImplementedError("This method should be overridden in a subclass")

class HumanPlayer(Player):
    def get_move(self, game_state):
        # Optionally print game state here
        move = input("Enter your move in hexadecimal format (e.g., 'A1'): ")
        x = int(move[0], 16)   # Convert the first part of the move to an integer
        y = int(move[1:], 16)  # Convert the second part of the move to an integer
        return x, y