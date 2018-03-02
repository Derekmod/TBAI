from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class HeuristicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = Variable(self.X.astype(float)[idx])
        x = Variable(self.Y.astype(float)[idx])
        return {'x': x, 'y': y}


def get_loader(dataset):
    return DataLoader(dataset, batch_size=4,
                      shuffle=False, num_workers=1)