import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tbai_nn import *




class ConnectFourNet(nn.Module):
    def __init__(self, pass_through_states=False, sideways_net=False, lam=0.):
        super(ConnectFourNet, self).__init__()

        self.pass_through = pass_through_states
        self.sideways = sideways_net
        self.lam = lam

        self.feature_size = 42
        self.state_size = 2

        self.conv1 = nn.Conv2d(1, 20, 3, padding=1, stride=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(20)
        #self.dropout1 = nn.Dropout2D(p=0.1)
        self.conv1_size = 180
        
        self.fc1_size = 80
        fc1_inp_size = self.conv1_size
        if pass_through_states:
            fc1_inp_size += self.state_size
        self.fc1 = nn.Linear(fc1_inp_size, self.fc1_size)

        self.fc2_size = 20
        fc2_inp_size = self.fc1_size
        if pass_through_states:
            fc2_inp_size += self.state_size
        self.fc2 = nn.Linear(fc2_inp_size, self.fc2_size)
        
        fco_inp_size = self.fc2_size
        if sideways_net:
            fco_inp_size += self.state_size + self.feature_size + self.conv1_size + self.fc1_size
        elif pass_through_states:
            fco_inp_size += self.state_size
        self.fc_out = nn.Linear(fco_inp_size, 2)

    def forward(self, bundle):
        self.feature_size = 42
        self.state_size = 2
        self.pass_through = True
        self.sideways = True
        position, state = bundle
        position = position.view(-1, 1, 7, 6)
        flat_position = position.view(-1, 42)
        state = state.view(-1, 2)

        conv_val = F.relu(F.max_pool2d(self.conv1(position), 2))
        conv_val = self.batch_norm1(conv_val)
        #conv_val = self.dropout(conv_val)
        flat_conv_val = conv_val.view(-1, self.conv1_size)

        fc1_inp = flat_conv_val
        if self.pass_through:
            fc1_inp = torch.cat((state, fc1_inp), 1)
        fc1_val = F.relu(self.fc1(fc1_inp))
        
        fc2_inp = fc1_val
        if self.pass_through:
            fc2_inp = torch.cat((state, fc2_inp), 1)
        fc2_val = F.relu(self.fc2(fc2_inp))

        final_inp = fc2_val
        if self.sideways:
            torch.cat((state, flat_position, flat_conv_val, fc1_val, final_inp), 1)
        elif self.pass_through:
            torch.cat((state, final_inp), 1)
        #final = F.sigmoid(self.fc_out(final_inp))
        final = gameActivation(self.fc_out(final_inp))

        return final