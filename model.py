import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42, fc1_units=256):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
