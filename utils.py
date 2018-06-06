import torch
import torch.nn.functional as F
from torch import nn
from torchnlp.text_encoders import WhitespaceEncoder, IdentityEncoder
from torchnlp.text_encoders.reserved_tokens import PADDING_TOKEN, UNKNOWN_TOKEN
from torchnlp.utils import datasets_iterator
from torchnlp.utils import pad_batch

from preprocessed_datasets import imdb_dataset, agnews_dataset, amazon_dataset, dbpedia_dataset, newsgroups_dataset, \
    reuters_dataset, webkb_dataset, yahoo_dataset, yelp_dataset


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


def load_data(data_type, fine_grained):
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

    sentence_corpus = [row['text'] for row in datasets_iterator(dataset[0], )]
    sentence_encoder = WhitespaceEncoder(sentence_corpus, reserved_tokens=[PADDING_TOKEN, UNKNOWN_TOKEN])
    label_corpus = [row['label'] for row in datasets_iterator(dataset[0], dataset[1])]
    label_encoder = IdentityEncoder(label_corpus, reserved_tokens=[])

    # Encode
    for row in datasets_iterator(dataset[0], dataset[1]):
        row['text'] = sentence_encoder.encode(row['text'])
        row['label'] = label_encoder.encode(row['label'])
    return sentence_encoder.vocab_size, label_encoder.vocab_size, dataset[0], dataset[1]


def collate_fn(batch):
    """ list of tensors to a batch tensors """
    text_batch, _ = pad_batch([row['text'] for row in batch])
    label_batch = [row['label'] for row in batch]
    return [text_batch, torch.cat(label_batch)]
