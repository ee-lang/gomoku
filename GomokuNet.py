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

class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        self.fc = nn.Linear(15 * 15 * 3, 15 * 15)  # Fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        q_values = self.fc(x)  # Compute Q-values
        return q_values.view(x.size(0), 15, 15)  # Reshape the output
