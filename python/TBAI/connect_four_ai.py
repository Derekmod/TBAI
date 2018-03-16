import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class ConnectFourNet(nn.Module):
    def __init__(self):
        super(ConnectFourNet, self).__init__()
        self.feature_size = 42
        self.state_size = 2

        self.conv1 = nn.Conv2d(1, 20, 3, padding=1, stride=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(20)
        #self.dropout1 = nn.Dropout2D(p=0.1)
        self.conv1_size = 180
        
        self.fc1_size = 80
        self.fc1 = nn.Linear(self.state_size + self.conv1_size, self.fc1_size)

        self.fc2_size = 20
        self.fc2 = nn.Linear(self.fc1_size + self.state_size, self.fc2_size)
        
        self.fc_out = nn.Linear(self.state_size + self.feature_size + self.conv1_size + self.fc1_size + self.fc2_size, 2)

    def forward(self, bundle):
        position, state = bundle
        position = position.view(-1, 1, 7, 6)
        print(position.shape)
        flat_position = position.view(-1, 42)
        state = state.view(-1, 2)

        conv_val = F.relu(F.max_pool2d(self.conv1(position), 2))
        conv_val = self.batch_norm1(conv_val)
        #conv_val = self.dropout(conv_val)
        flat_conv_val = conv_val.view(-1, self.conv1_size)

        fc1_inp = torch.cat((state, flat_conv_val), 0)
        fc1_val = F.relu(self.fc1(fc1_inp))
        
        fc2_inp = torch.cat((state, fc1_val), 0)
        fc2_val = F.relu(self.fc2(fc2_inp))

        final_inp = torch.cat((state, flat_position, flat_conv_val, fc1_val, fc2_val), 0)
        final = F.sigmoid(self.fc_out(final_inp))

        return final