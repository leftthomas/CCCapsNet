import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchnlp.text_encoders import WhitespaceEncoder

from datasets import imdb_dataset, agnews_dataset, amazon_dataset, dbpedia_dataset, newsgroups_dataset, \
    reuters_dataset, webkb_dataset, yahoo_dataset, yelp_dataset


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


def load_data(data_type, batch_size, fine_grained):
    if data_type == 'imdb':
        dataset = imdb_dataset(train=True, test=True)
    elif data_type == 'newsgroups':
        dataset = newsgroups_dataset(train=True, test=True)
    elif data_type == 'reuters':
        dataset = reuters_dataset(train=True, test=True, fine_grained=fine_grained)
    elif data_type == 'webkb':
        dataset = webkb_dataset(train=True, test=True)
    elif data_type == 'dbpedia':
        dataset = dbpedia_dataset(train=True, test=True)
    elif data_type == 'agnews':
        dataset = agnews_dataset(train=True, test=True)
    elif data_type == 'yahoo':
        dataset = yahoo_dataset(train=True, test=True)
    elif data_type == 'yelp':
        dataset = yelp_dataset(train=True, test=True, fine_grained=fine_grained)
    elif data_type == 'amazon':
        dataset = amazon_dataset(train=True, test=True, fine_grained=fine_grained)
    else:
        raise ValueError('{} data type not supported.'.format(data_type))

    encoder = WhitespaceEncoder(dataset[0].__getitem__('text'), append_eos=True)
    labels = list(set(dataset[0].__getitem__('label')))
    return encoder, labels, DataLoader(dataset[0], batch_size, shuffle=True, num_workers=4), DataLoader(dataset[1],
                                                                                                        batch_size,
                                                                                                        num_workers=4)
