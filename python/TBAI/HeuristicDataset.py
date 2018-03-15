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
    #print('batch[0]', batch[0])
    k = len(batch[0])
    trans = [[] for _ in range(k)]
    for el in batch:
        for i in range(k):
            trans[i] += [el[i]]

    ret = []
    for feat_batch in trans:
        #print('feature batch', feat_batch)
        ret += [ torch.stack(feat_batch, 0) ]
    #print('new ret', ret)
    return ret

def TBAI_collate(batch):
    X = tuple_collate([sample['x'] for sample in batch])
    Y = torch.stack([sample['y'] for sample in batch], 0)

    #print('x shapes:', [x.shape for x in X])
    #print(Y.shape)

    return {'x':X, 'y':Y}