import torch.nn.functional as F
from torch import nn
from torchnlp.datasets import imdb_dataset, smt_dataset, trec_dataset


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


def load_data(data_type, train_mode, batch_size, fine_grained):
    if data_type == 'TREC':
        dataset = trec_dataset(train=train_mode, test=not train_mode, fine_grained=fine_grained)
    elif data_type == 'SMT':
        dataset = smt_dataset(train=train_mode, test=not train_mode, fine_grained=fine_grained)
    elif data_type == 'IMDB':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'Newsgroups':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'Reuters':
        dataset = imdb_dataset(train=train_mode, test=not train_mode, fine_grained=fine_grained)
    elif data_type == 'Cade':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'WebKB':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'DBPedia':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'YahooAnswers':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'SogouNews':
        dataset = imdb_dataset(train=train_mode, test=not train_mode)
    elif data_type == 'YelpReview':
        dataset = imdb_dataset(train=train_mode, test=not train_mode, fine_grained=fine_grained)
    elif data_type == 'AmazonReview':
        dataset = imdb_dataset(train=train_mode, test=not train_mode, fine_grained=fine_grained)
