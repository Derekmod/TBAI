from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch


class HeuristicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        #x = Variable(x)

        y = self.Y[idx]
        y = torch.from_numpy(y)
        y = y.type(torch.FloatTensor)
        #y = Variable(y)
        return {'x': x, 'y': y}


def get_loader(dataset):
    return DataLoader(dataset, batch_size=4,
                      shuffle=False, num_workers=1)