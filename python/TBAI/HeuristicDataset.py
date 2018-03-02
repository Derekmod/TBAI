from torch.utils.data import Dataset, DataLoader


class HeuristicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.Y[idx]}


def get_loader(dataset):
    return DataLoader(dataset, batch_size=4,
                      shuffle=False, num_workers=1)