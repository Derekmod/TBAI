from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

from torch.utils.data.dataloader import default_collate


class HeuristicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        #x = torch.from_numpy(x)
        #x = x.type(torch.FloatTensor)
        #x = Variable(x)

        y = self.Y[idx]
        #y = torch.from_numpy(y)
        #y = y.type(torch.FloatTensor)
        #y = Variable(y)
        return {'x': x, 'y': y}


def get_loader(dataset):
    return DataLoader(dataset, batch_size=4,
                      shuffle=False, num_workers=1,
                      collate_fn=TBAI_collate)

def tuple_collate(batch):
    k = len(batch[0])
    trans = [[] for _ in range(k)]
    for el in batch:
        for i in range(k):
            trans[i] += [el[i]]

    ret = []
    for feat_batch in trans:
        ret += [ torch.stack(feat_batch, 0) ]
    return ret

def TBAI_collate(batch):
    ''' Collates multiple samples into one batch.
    Args:
        batch {string --> [FloatTensor]}:
            batch['x'] [[FloatTensor]]: feature list
            batch['y'] [FloatTensor]: labels
    Returns:
        ret['x'] [[FloatTensor]] list of features, each feature stacked on 0th dimension
        ret['y'] [FloatTensor] labels stacked on 0th dimension
    '''
    X = tuple_collate([sample['x'] for sample in batch])
    Y = torch.stack([sample['y'] for sample in batch], 0)

    return {'x':X, 'y':Y}