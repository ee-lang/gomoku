class Player:
    def get_move(self, game_state):
        raise NotImplementedError("This method should be overridden in a subclass")
    def score(self, score):
        raise NotImplementedError("This method should be overridden in a subclass if necessary")

class HumanPlayer(Player):
    def get_move(self, game_state):
        # Optionally print game state here
        move = input("Enter your move in hexadecimal format (e.g., 'A1'): ")
        x = int(move[0], 16)   # Convert the first part of the move to an integer
        y = int(move[1:], 16)  # Convert the second part of the move to an integer
        return x, y

    def score(self, score):
        print(f"Your score: {score}")

import random

class RandomPlayer(Player):
    def get_move(self, game_state):
        # Get the current board from the game state
        board = game_state["board"]
        
        # Get the list of available moves (empty spots on the board)
        available_moves = [(i, j) for i, row in enumerate(board) for j, spot in enumerate(row) if spot == ' ']

        # Choose a random move from the list of available moves
        move = random.choice(available_moves)

        return move

    def score(self, score):
        print(f"Random player score: {score}")

# GPT4 generated QLearning player

import numpy as np
from collections import defaultdict
import pickle

class QLearningPlayer(Player):
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.last_state = None
        self.last_action = None

    def get_move(self, game_state):
        # Convert the game_state's board to a tuple of tuples
        state = tuple(map(tuple, game_state['board']))

        # Get the available moves
        available_moves = self.available_moves(state)

        # Choose an action
        if np.random.rand() < self.epsilon:  # exploration
            action = random.choice(available_moves)
        else:  # exploitation
            q_values = [self.q_table[(state, action)] for action in available_moves]
            action = available_moves[np.argmax(q_values)]

        # If this is not the first move, update the Q-value of the last state-action pair
        if self.last_state is not None and self.last_action is not None:
            self.update_q_value(0, self.last_state, self.last_action, state)

        # Store the current state and action for the next time
        self.last_state = state
        self.last_action = action

        return action

    def update_q_value(self, reward, old_state, old_action, new_state):
        old_q_value = self.q_table[(old_state, old_action)]
        future_q_values = None
        if new_state is not None:
            future_q_values = [self.q_table[(new_state, action)] for action in self.available_moves(new_state)]
        if future_q_values:
            max_future_q_value = max(future_q_values)
        else:
            max_future_q_value = 0

        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q_value - old_q_value)
        self.q_table[(old_state, old_action)] = new_q_value

    def score(self, score):
        # Update Q-value with the final reward
        # Using different values as reward i.e 1 for win, 0.5 for draw, and -1 for loss
        reward = 2 * score - 1.0
        self.update_q_value(reward, self.last_state, self.last_action, None)

        # Reset the last state and action for the next game
        self.last_state = None
        self.last_action = None
        self.show_qvalues()

    def available_moves(self, game_state):
        return [(i, j) for i, row in enumerate(game_state) for j, spot in enumerate(row) if spot == ' ']

    def show_qvalues(self):
        for k,v in self.q_table.items():
            print('k,v',k,v)
    
    def save_q_table(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)        

    def load_q_table(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
        except FileNotFoundError:
            print(f'{file_path} not found. Using empty values instead.')
            self.q_table = defaultdict(float)


import torch
import torch.nn.functional as F

class QNNLearningPlayer(Player):
    def __init__(self, model, optimizer, device='cpu', gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.model = model  # the network model
        self.optimizer = optimizer  # the optimizer
        self.device = device  # 'cpu' or 'cuda'
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # initial exploration rate
        self.epsilon_end = epsilon_end  # final exploration rate
        self.epsilon_decay = epsilon_decay  # rate of decay of epsilon per episode

        self.model.to(self.device)
        self.last_state = None
        self.last_action = None


    def get_move(self, game_state):
        # If this is not the first move, update the Q-values based on the last state and action
        if self.last_state is not None and self.last_action is not None:
            # Assume a reward of 0 for non-terminal states
            self.update(self.last_state, self.last_action, 0, game_state)

        # Convert the game state to a tensor
        state_tensor = self._preprocess(game_state).unsqueeze(0).to(self.device)

        # Choose the action
        if np.random.rand() < self.epsilon:  # exploration
            # action = np.random.choice(self.available_moves(game_state))
            action = random.choice(self.available_moves(game_state))
        else:  # exploitation
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
                action = np.unravel_index(q_values.argmax(), q_values.shape)

        # Store the current state and action for the next update
        self.last_state = game_state
        self.last_action = action

        return action

    def score(self, score):
        # Update the Q-values based on the final state and reward
        if self.last_state is not None and self.last_action is not None:
            # Using different values as reward i.e 1 for win 0.5 for draw and -1 for lose
            score = 2 * score - 1.0
            self.update(self.last_state, self.last_action, score, None)

        # Decay the exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def update(self, state, action, reward, next_state):
        # Convert to tensors
        state = self._preprocess(state).unsqueeze(0).to(self.device)
        next_state = self._preprocess(next_state).unsqueeze(0).to(self.device) if next_state is not None else None
        reward = torch.tensor([reward], device=self.device)
        action = torch.tensor([action], device=self.device)

        # Compute Q(s, a)
        q_values = self.model(state)
        x,y = action[0][0], action[0][1]
        # q_value = q_values.squeeze(0)[action[0]][action[1]]
        q_value = q_values.squeeze(0)[x][y]

        # Compute the target value
        if next_state is None:  # if this is the final state
            target = reward
        else:
            with torch.no_grad():
                next_q_values = self.model(next_state)
                next_q_value = next_q_values.max()
            target = reward + self.gamma * next_q_value

        # Update the model
        loss = F.mse_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _preprocess(self, game_state):
        # Convert the game state to a tensor
        state = np.zeros((3, 15, 15), dtype=np.float32)
        for i in range(15):
            for j in range(15):
                if game_state['board'][i][j] == 'X':
                    state[0, i, j] = 1
                elif game_state['board'][i][j] == 'O':
                    state[1, i, j] = 1
                else:
                    state[2, i, j] = 1
        return torch.from_numpy(state)

    def available_moves(self, game_state):
        # Get the list of available moves (empty spots on the board)
        return [(i, j) for i, row in enumerate(game_state['board']) for j, spot in enumerate(row) if spot == ' ']

    def end_episode(self):
        # Decay the exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
