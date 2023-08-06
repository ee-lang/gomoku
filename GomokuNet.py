import torch
import torch.nn as nn
import torch.nn.functional as F

# class GomokuNet(nn.Module):
#     def __init__(self):
#         super(GomokuNet, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
#         # Final layer to produce Q-values
#         self.out = nn.Conv2d(128, 1, kernel_size=1)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         q_values = self.out(x)
#         return q_values

# Linear fully connected 1 hidden layer neural net
class GomokuLFC1HNNet(nn.Module):
    def __init__(self, input_size=3*15*15+1, hidden_size=128, output_size=15*15):
        super(GomokuLFC1HNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float32)  # Fully connected layer
        self.fc_h = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)  # Fully connected hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float32)  # Fully connected hidden layer
        # self.fc = nn.Linear()  # Fully connected layer

    def forward(self, x, action_mask = None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_h(x))
        x = self.fc2(x)  # Compute Q-values
        # print('action_mask',action_mask)
        # x = torch.clip(x, min=-1.0, max=1.0)
        return torch.where(action_mask, x, torch.tensor(float('-inf'))) if action_mask is not None else x
        # return x.view(x.size(0), output_size)  # Reshape the output

    # def forward(self, x, action_mask):
    #     q_values = self.fc(x)
    #     masked_q_values = torch.where(action_mask, q_values, torch.tensor(float('-inf')))
    #     return masked_q_values