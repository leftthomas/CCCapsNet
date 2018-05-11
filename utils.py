import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
from torch import nn


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


def _iters(data_type, batch_size, fine_grained):
    text = data.Field(sequential=True)
    label = data.LabelField()

    train, val, test = datasets.SST.splits(text, label, root='data', fine_grained=fine_grained)

    text.build_vocab(train, vectors='glove.6B.300d')
    label.build_vocab(train)

    return text, label, data.BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, device=-1)


def load_data(data_type, batch_size, fine_grained=True):
    text, label, (train_iter, val_iter, test_iter) = _iters(data_type, batch_size, fine_grained)

    data_info = {'vocab_size': len(text.vocab), 'num_class': 5 if fine_grained else 3, 'text': text}

    return BatchWrapper(train_iter), BatchWrapper(val_iter), BatchWrapper(test_iter), data_info
