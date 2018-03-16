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
        ''' Heuristic function.
        Args:
            bundle (Variable):
                position <FloatTensor>: any x 3x3, board position
                state_vals <FloatTensor>: any x 2, player_turn and turn_idx
        Returns:
            <FloatTensor> any x 2: each group has values:
                probability of winning
                expected error
        '''
        x, state_vals = bundle
        flat_x = x.view(-1, self.position_size)
        state_vals = state_vals.view(-1, self.state_size)

        inp1 = torch.cat((state_vals, flat_x), 1)
        out1 = F.relu(self.fc1(inp1))

        inp2 = torch.cat((state_vals, out1), 1)
        out2 = F.relu(self.fc2(inp2))

        inp3 = torch.cat((state_vals, out1, out2), 1)
        out3 = F.relu(self.fc3(inp3))

        y = F.sigmoid(out3)
        return y