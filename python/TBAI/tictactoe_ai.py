import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.position_size = 9
        self.state_size = 2
        self.fc1_size = 50
        self.fc2_size = 20

        self.fc1 = nn.Linear(self.position_size + self.state_size, self.fc1_size)
        self.fc2 = nn.Linear(self.state_size + self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.state_size + self.fc1_size + self.fc2_size, 2)

    def forward(self, bundle):
        x, state_vals = bundle
        flat_x = x.view(-1, self.position_size)
        state_vals = state_vals.view(-1, self.state_size)
        print('x shape:', x.shape)
        print('flat_x shape:', flat_x.shape)
        print('state_vals shape:', state_vals.shape)

        inp1 = torch.cat((state_vals, flat_x), 1)
        out1 = F.relu(self.fc1(inp1))

        inp2 = torch.cat((state_vals, out1), 1)
        out2 = F.relu(self.fc2(inp2))

        inp3 = torch.cat((state_vals, out1, out2), 1)
        out3 = F.relu(self.fc3(inp3))

        y = F.sigmoid(out3)
        return y