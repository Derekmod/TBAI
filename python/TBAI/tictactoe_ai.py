import torch
import torch.nn as nn
import torch.nn.functional as F




class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(51, 20)
        self.fc3 = nn.Linear(21, 2)
        
        self.fc_out = nn.Linear(1+9+50+20, 2)

    def forward(self, x):
        turn = x[0:1]
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(turn + x1))
        y = F.sigmoid(self.fc3(turn + x2))
        
        # y = F.sigmoid(self.fc_out(turn + x + x1 + x2))
        
        return y
        
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)