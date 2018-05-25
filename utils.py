import torch.nn.functional as F
import torchtext.data as data
from torch import nn
from torchtext.datasets import SST, TREC, IMDB


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


class BatchWrapper:
    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for batch in self.dl:
            text = getattr(batch, 'text')
            label = getattr(batch, 'label')
            yield [text, label]

    def __len__(self):
        return len(self.dl)


def load_data(data_type, batch_size, fine_grained):
    text = data.Field(sequential=True, lower=True, batch_first=True)
    label = data.LabelField(batch_first=True)

    if data_type == 'TREC':
        train, test = TREC.splits(text, label, root='data', fine_grained=fine_grained)
    elif data_type == 'SST':
        train, val, test = SST.splits(text, label, root='data', fine_grained=fine_grained)
    else:
        # IMDB
        train, test = IMDB.splits(text, label, root='data')

    text.build_vocab(train)
    label.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, repeat=False, device=-1)
    data_info = {'vocab_size': len(text.vocab), 'num_class': len(label.vocab)}

    return BatchWrapper(train_iter), BatchWrapper(test_iter), data_info
