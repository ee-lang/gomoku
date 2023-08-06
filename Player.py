class Player:
    def get_move(self, game_state):
        raise NotImplementedError("This method should be overridden in a subclass")
    def score(self, score):
        raise NotImplementedError("This method should be overridden in a subclass if necessary")

    def log_board(self, logger, board):
        """
        Print the current state of the game board. Empty spots are represented by ' ',
        and players' pieces are represented by 'X' and 'O'.
        """
        # Print the column numbers
        logger.debug('Board')
        # Print the board with row numbers
        for i, row in enumerate(board):
            logger.debug(' '.join(row))


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
        pass
        # print(f"Random player score: {score}")

# GPT4 generated QLearning player

import numpy as np
from collections import defaultdict
import pickle
import logging

class QLearningPlayer(Player):
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, training = False, logger = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.last_state = None
        self.last_action = None
        self.training = training
        self.logger = self.logger = logger if logger else logging.getLogger('gomoku_player')

    def get_move(self, game_state):
        # Convert the game_state's board to a tuple of tuples
        state = tuple(map(tuple, game_state['board']))

        # Get the available moves
        available_moves = self.available_moves(state)

        # Choose an action
        if np.random.rand() < self.epsilon:  # exploration
            self.logger.debug('random choice')
            action = random.choice(available_moves)
        else:  # exploitation
            self.logger.debug('exploitation')
            self.logger.debug(f'state {state}')
            self.logger.debug(f'available_moves {available_moves}')
            q_values = [self.q_table[(state, action)] for action in available_moves]
            self.logger.debug(f'q_values for state {q_values}')
            action = available_moves[np.argmax(q_values)]

        if self.training:
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
        if self.training:
            # Update Q-value with the final reward
            # Using different values as reward i.e 1 for win, 0.5 for draw, and -1 for loss
            reward = 2 * score - 1.0
            self.update_q_value(reward, self.last_state, self.last_action, None)

            # Reset the last state and action for the next game
            self.last_state = None
            self.last_action = None
            # self.show_qvalues()

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
import copy
import random

class QNNLearningPlayer(Player):
    def __init__(self, model, optimizer, board_size=15, device='cpu', alpha=0.5, gamma=0.9, epsilon_start=0.5, epsilon_end=1e-4, epsilon_decay=0.99, training=False, logger=None, batch_size=10):
        self.model = model  # the network model
        self.optimizer = optimizer  # the optimizer
        self.device = device  # 'cpu' or 'cuda'
        self.gamma = gamma  # discount factor
        self.alpha = alpha
        self.epsilon = epsilon_start  # initial exploration rate
        self.epsilon_end = epsilon_end  # final exploration rate
        self.epsilon_decay = epsilon_decay  # rate of decay of epsilon per episode
        self.board_size = board_size

        self.model.to(self.device)
        self.last_state = None
        self.last_action = None
        self.training = training
        self.logger = self.logger = logger if logger else logging.getLogger('gomoku_player')
        self.transitions = []
        self.prio_transitions = []
        self.batch_size = batch_size


    def get_move(self, game_state):

        # Convert the game state to a tensor
        state_tensor = self._preprocess(game_state).to(self.device)
        state_id = int(''.join(map(str, state_tensor.int().tolist())), 2)

        available_moves = self.available_moves(game_state)
        
        valid_action = False

        self.log_board(self.logger, game_state['board'])
        self.logger.debug(f'state id={state_id}')

        while not valid_action:
            # Choose the action
            if np.random.rand() < self.epsilon:  # exploration
                # action = np.random.choice(self.available_moves(game_state))
                self.logger.debug('random choice')
                action = random.choice(available_moves)
                valid_action = True
            else:  # exploitation
                self.logger.debug('exploitation')
                self.logger.debug(f'state {state_tensor}')
                self.logger.debug(f'available_moves {available_moves}')
                with torch.no_grad():
                    action_mask = torch.tensor([1 if (i,j) in available_moves else 0 for i in range(self.board_size) for j in range(self.board_size)], dtype=torch.bool)
                    q_values = self.model(state_tensor, action_mask).squeeze(0).cpu().numpy()
                    q_values = q_values.reshape((self.board_size, self.board_size))
                    action = np.unravel_index(q_values.argmax(), q_values.shape)
                    self.logger.debug(f'q_values estimates {q_values}')
                    if action in available_moves:
                        self.logger.debug(f'Valid action {action}')
                        valid_action = True
                        continue
                self.logger.debug(f'Invalid action {action}')
                # self.update(game_state, action, q_values.min(), None)

        if self.training:
            # If this is not the first move, update the Q-values based on the last state and action
            if self.last_state is not None and self.last_action is not None:
                # Assume a reward of 0 for non-terminal states
                self.update(self.last_state, self.last_action, 0, game_state)

            # Store the current state and action for the next update
            self.last_state = copy.deepcopy(game_state)
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
        state_tensor = self._preprocess(state).to(self.device)
        next_available_moves = self.available_moves(next_state) if next_state is not None else None
        next_state_tensor = self._preprocess(next_state).to(self.device) if next_state is not None else None
        reward = torch.tensor(reward, device=self.device)
        action = torch.tensor(action, device=self.device)

        state_id = int(''.join(map(str, state_tensor.int().tolist())), 2)


        # Store transition to experience memory
        self.transitions.append((state_tensor, action, reward, next_state_tensor, next_available_moves, state_id))
        
        # Only run a learning phase when an episode ends
        if next_state_tensor is None:
            self.prio_transitions.append((state_tensor, action, reward, next_state_tensor, next_available_moves, state_id))

            # Replay sample
            sample_batch = random.sample(self.transitions, self.batch_size) if len(self.transitions)>self.batch_size else self.transitions

            prio_sample_batch = random.sample(self.prio_transitions, self.batch_size) if len(self.prio_transitions)>self.batch_size else self.prio_transitions        

            for state_tensor, action, reward, next_state_tensor, next_available_moves, state_id in sample_batch + prio_sample_batch:
                self.logger.debug(f'sample (state_tensor={state_tensor}, action={action}, reward={reward}, next_state_tensor={next_state_tensor}, state_id={state_id})')
                # Compute Q(s, a)
                q_values = self.model(state_tensor)
                y,x = action[0], action[1]
                i = y*self.board_size+x
                q_value = q_values.squeeze(0)[i]
                
                target = q_values.clone().detach()

                # Compute the target value
                if next_state_tensor is None:  # if this is the final state
                    target[i] = reward
                    # target = reward
                else:
                    with torch.no_grad():
                        action_mask = torch.tensor([1 if (i,j) in next_available_moves else 0 for i in range(self.board_size) for j in range(self.board_size)], dtype=torch.bool)
                        next_q_values = self.model(next_state_tensor, action_mask)
                        next_q_value = next_q_values.max()
                    target[i] = reward + self.gamma * next_q_value
                    # target[i] = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
                    # target = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)

                # Update the model
                self.optimizer.zero_grad()
                # self.logger.debug(f'loss fn, i={i} q_values={q_values}, target={target}, delta={target[i]-q_values[i]}')
                # if abs(reward)>0.1:
                #     self.logger.debug(f'state={state_tensor}\nreward={reward} i={i} q_values={q_values}, q_value={q_value}, target={target}, delta={target-q_values}')
                loss = F.mse_loss(q_values, target)
                # loss = F.mse_loss(q_value, target)
                loss.backward()
                # for p in self.model.parameters():
                #     self.logger.debug(f'param shape {p.shape} {p}\ngrad {p.grad}')
                self.optimizer.step()
                # if abs(reward)>0.1:
                #     new_q_values = self.model(state_tensor)        
                #     self.logger.debug(f'After optimization\nq_values={new_q_values}')

                # for p in self.model.parameters():
                #     self.logger.debug(f'param shape {p.shape} {p}')

    def _preprocess(self, game_state):
        board_size = len(game_state['board'])
        # Convert the game state to a tensor
        state = np.zeros((3*board_size*board_size+1), dtype=np.float32)
        offset = board_size*board_size
        for i in range(board_size):
            for j in range(board_size):
                if game_state['board'][i][j] == 'X':
                    state[i*board_size + j] = 1
                elif game_state['board'][i][j] == 'O':
                    state[offset+i*board_size + j] = 1
                else:
                    state[2*offset+i*board_size + j] = 1
        # information about the current player
        if game_state['current_player'] == 'O':
            state[3*offset] = 1
        return torch.from_numpy(state)

    def available_moves(self, game_state):
        # Get the list of available moves (empty spots on the board)
        return [(i, j) for i, row in enumerate(game_state['board']) for j, spot in enumerate(row) if spot == ' ']

    def end_episode(self):
        # Decay the exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def load_terminal_transactions(self, file_path='t_txn.pkl'):
        try:
            with open(file_path, 'rb') as f:
                self.prio_transitions = pickle.load(f)
                print('len(self.prio_transitions)',len(self.prio_transitions))
        except FileNotFoundError:
            print(f'{file_path} not found. Using empty values instead.')
            self.prio_transitions = []

    def save_terminal_transactions(self, file_path='t_txn.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(self.prio_transitions, f)        


    def offline_training(self, epochs=1, batch_size=10, max_iter=100, eps=1e-4):
        for e in range(1, epochs+1):
            # Replay sample
            prio_sample_batch = random.sample(self.prio_transitions, batch_size) if len(self.prio_transitions)>batch_size else self.prio_transitions        
            losses = []

            for state_tensor, action, reward, next_state_tensor, next_available_moves, state_id in prio_sample_batch:
                self.logger.debug(f'sample (state_tensor={state_tensor}, action={action}, reward={reward}, next_state_tensor={next_state_tensor}, state_id={state_id})')
                for c in range(max_iter):
                    # Compute Q(s, a)
                    q_values = self.model(state_tensor)
                    self.logger.debug(f'q_values={q_values}')
                    y,x = action[0], action[1]
                    i = y*self.board_size+x
                    q_value = q_values.squeeze(0)[i]

                    target = q_values.clone().detach()

                    # Compute the target value
                    if next_state_tensor is None:  # if this is the final state
                        target[i] = reward
                    else:
                        with torch.no_grad():
                            action_mask = torch.tensor([1 if (i,j) in next_available_moves else 0 for i in range(self.board_size) for j in range(self.board_size)], dtype=torch.bool)
                            next_q_values = self.model(next_state_tensor, action_mask)
                            next_q_value = next_q_values.max()
                        target[i] = reward + self.gamma * next_q_value

                    # Update the model
                    self.optimizer.zero_grad()
                    loss = F.mse_loss(q_values, target)
                    losses.append(loss.item())
                    self.logger.debug(f'sample iter={c} MSE-loss={loss.item()}')
                    # loss = F.mse_loss(q_value, target)
                    loss.backward()
                    # for p in self.model.parameters():
                    #     self.logger.debug(f'param shape {p.shape} {p}\ngrad {p.grad}')
                    self.optimizer.step()
                    if loss.item() < eps:
                        q_values = self.model(state_tensor)
                        self.logger.debug(f'q_values after optimisation={q_values}')
                        break

    def eval_model_on_terminal_txn(self, sample_size=1000):
            prio_sample_batch = random.sample(self.prio_transitions, sample_size) if len(self.prio_transitions)>sample_size else self.prio_transitions        
            losses = []

            for state_tensor, action, reward, next_state_tensor, next_available_moves, state_id in prio_sample_batch:
                # self.logger.debug(f'sample (state_tensor={state_tensor}, action={action}, reward={reward}, next_state_tensor={next_state_tensor}, state_id={state_id})')
                with torch.no_grad():
                    # Compute Q(s)
                    q_values = self.model(state_tensor)
                    # self.logger.debug(f'q_values={q_values}')
                    y,x = action[0], action[1]
                    i = y*self.board_size+x
                    q_value = q_values.squeeze(0)[i]

                    target = q_values.clone().detach()

                    # Compute the target value
                    if next_state_tensor is None:  # if this is the final state
                        target[i] = reward
                    else:
                        action_mask = torch.tensor([1 if (i,j) in next_available_moves else 0 for i in range(self.board_size) for j in range(self.board_size)], dtype=torch.bool)
                        next_q_values = self.model(next_state_tensor, action_mask)
                        next_q_value = next_q_values.max()
                        target[i] = reward + self.gamma * next_q_value

                    # Update the model
                    loss = F.mse_loss(q_values, target)
                    losses.append(loss.item())
                    self.logger.debug(f'sample state_id={state_id}, action={action}, reward={reward}, MSE-loss={loss.item()}')
                    # q_values = self.model(state_tensor)
                    # self.logger.debug(f'q_values after optimisation={q_values}')

            print(f'Final MSE-loss={np.mean(losses)}')

