import torch.nn.functional as F
from torch import nn
from torchnlp.datasets import imdb_dataset, smt_dataset, trec_dataset
from torchnlp.utils import datasets_iterator

from datasets import ag_dataset, amazon_dataset, cade_dataset, dbpedia_dataset, newsgroups_dataset, reuters_dataset, \
    sogou_dataset, webkb_dataset, yahoo_dataset, yelp_dataset


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
        dataset = trec_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode,
                               fine_grained=fine_grained)
    elif data_type == 'SMT':
        dataset = smt_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode,
                              fine_grained=fine_grained)
    elif data_type == 'IMDB':
        dataset = imdb_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'Newsgroups':
        dataset = newsgroups_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'Reuters':
        dataset = reuters_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode,
                                  fine_grained=fine_grained)
    elif data_type == 'Cade':
        dataset = cade_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'WebKB':
        dataset = webkb_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'DBPedia':
        dataset = dbpedia_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'AGNews':
        dataset = ag_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'YahooAnswers':
        dataset = yahoo_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'SogouNews':
        dataset = sogou_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode)
    elif data_type == 'YelpReview':
        dataset = yelp_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode,
                               fine_grained=fine_grained)
    elif data_type == 'AmazonReview':
        dataset = amazon_dataset(directory='data/' + data_type, train=train_mode, test=not train_mode,
                                 fine_grained=fine_grained)
    else:
        raise ValueError('Expected data_type must be one of TREC, SMT, IMDB, Newsgroups, Reuters, Cade, WebKB, DBPedia,'
                         ' AGNews, YahooAnswers, SogouNews, YelpReview and AmazonReview, got {} instead.'
                         .format(data_type))

    return datasets_iterator([dataset])
