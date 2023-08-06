# from Player import Player, HumanPlayer, RandomPlayer, QLearningPlayer, QNNLearningPlayer  # assuming player.py is the file containing the Player classes
from GomokuNet import GomokuLFC1HNNet
# import logging
import torch
import sys

model=GomokuLFC1HNNet(input_size=3*3*3+1, hidden_size=32, output_size=9)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# Load the saved weights into the model
try:
    model.load_state_dict(torch.load('GomokuLFC1HNNet_p1.pth'))
except FileNotFoundError as e:
    print(e)

# player1 = QNNLearningPlayer(model=model, optimizer=optimizer, board_size=3, training=False, logger=logger)

def tensor_to_board(input_tensor, b_size=3):
    s = ""
    for i in range(b_size):
        for j in range(b_size):
            s += 'X' if input_tensor[i*b_size+j]==1.0 else 'O' if input_tensor[b_size**2+i*b_size+j]==1.0 else '_'
            s += ' '
        s+='\n'
    return s

# 2023-08-06 12:45:21,880 - gomoku_training_logger - DEBUG - sample (state_tensor=tensor([0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
#         0., 0., 1., 0., 0., 0., 1., 0., 1., 0.]), action=tensor([2, 0]), reward=-1.0, next_state_tensor=None, state_id=14057610)

def int_to_binary_tensor(integer_value, length):
    binary_str = format(integer_value, f'0{length}b')
    binary_list = [int(bit) for bit in binary_str]
    binary_tensor = torch.tensor(binary_list, dtype=torch.float32)
    return binary_tensor


# Example usage:
integer_value = 135315866
length = 28

if len(sys.argv) > 1:
    # Get the integer argument
    integer_value = int(sys.argv[1])

binary_tensor = int_to_binary_tensor(integer_value, length)

with torch.no_grad():
    # state_tensor = torch.tensor([0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 1., 1., 0., 1., 1., 0., 1., 0.], dtype=torch.float32)

    # state_tensor = torch.tensor([0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.])
    # print(int(''.join(map(str, state_tensor.int().tolist())), 2))
    # print(integer_value)
    state_tensor = int_to_binary_tensor(integer_value, length)
    print(state_tensor)
    # print(int(''.join(map(str, state_tensor.int().tolist())), 2))
    print(tensor_to_board(state_tensor))
    print(model(state_tensor).view((3,3)))


