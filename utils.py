import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchnlp.text_encoders import WhitespaceEncoder, IdentityEncoder
from torchnlp.text_encoders.reserved_tokens import PADDING_TOKEN, UNKNOWN_TOKEN
from torchnlp.utils import datasets_iterator
from torchnlp.utils import pad_batch

from datasets import imdb_dataset, agnews_dataset, amazon_dataset, dbpedia_dataset, newsgroups_dataset, reuters_dataset, \
    webkb_dataset, yahoo_dataset, yelp_dataset, cade_dataset, sogou_dataset


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


def load_data(data_type, preprocessing=False, fine_grained=False, verbose=False, text_length=1200, encode=True):
    if data_type == 'imdb':
        train_data, test_data = imdb_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'newsgroups':
        train_data, test_data = newsgroups_dataset(preprocessing=preprocessing, verbose=verbose,
                                                   text_length=text_length)
    elif data_type == 'reuters':
        train_data, test_data = reuters_dataset(preprocessing=preprocessing, fine_grained=fine_grained, verbose=verbose,
                                                text_length=text_length)
    elif data_type == 'webkb':
        train_data, test_data = webkb_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'cade':
        train_data, test_data = cade_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'dbpedia':
        train_data, test_data = dbpedia_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'agnews':
        train_data, test_data = agnews_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'yahoo':
        train_data, test_data = yahoo_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'sogou':
        train_data, test_data = sogou_dataset(preprocessing=preprocessing, verbose=verbose, text_length=text_length)
    elif data_type == 'yelp':
        train_data, test_data = yelp_dataset(preprocessing=preprocessing, fine_grained=fine_grained, verbose=verbose,
                                             text_length=text_length)
    elif data_type == 'amazon':
        train_data, test_data = amazon_dataset(preprocessing=preprocessing, fine_grained=fine_grained, verbose=verbose,
                                               text_length=text_length)
    else:
        raise ValueError('{} data type not supported.'.format(data_type))

    if encode:
        sentence_corpus = [row['text'] for row in datasets_iterator(train_data, )]
        sentence_encoder = WhitespaceEncoder(sentence_corpus, reserved_tokens=[PADDING_TOKEN, UNKNOWN_TOKEN])
        label_corpus = [row['label'] for row in datasets_iterator(train_data, )]
        label_encoder = IdentityEncoder(label_corpus, reserved_tokens=[])

        # Encode
        for row in datasets_iterator(train_data, test_data):
            row['text'] = sentence_encoder.encode(row['text'])
            row['label'] = label_encoder.encode(row['label'])
        return sentence_encoder.vocab_size, label_encoder.vocab_size, train_data, test_data
    else:
        return train_data, test_data


def collate_fn(batch):
    """ list of tensors to a batch tensors """
    text_batch, _ = pad_batch([row['text'] for row in batch])
    label_batch = [row['label'] for row in batch]
    return [text_batch, torch.cat(label_batch)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Preprocessed Data')
    parser.add_argument('--data_type', default='imdb', type=str,
                        choices=['imdb', 'newsgroups', 'reuters', 'webkb', 'cade', 'dbpedia', 'agnews', 'yahoo',
                                 'sogou', 'yelp', 'amazon'], help='dataset type')
    parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                    'reuters, yelp and amazon')
    opt = parser.parse_args()
    DATA_TYPE = opt.data_type
    FINE_GRAINED = opt.fine_grained
    train_dataset, test_dataset = load_data(DATA_TYPE, preprocessing=None, fine_grained=FINE_GRAINED, encode=False)

    if FINE_GRAINED and DATA_TYPE in ['reuters', 'yelp', 'amazon']:
        train_file = os.path.join('data', DATA_TYPE, 'preprocessed_fine_grained_train.csv')
        test_file = os.path.join('data', DATA_TYPE, 'preprocessed_fine_grained_test.csv')
    else:
        train_file = os.path.join('data', DATA_TYPE, 'preprocessed_train.csv')
        test_file = os.path.join('data', DATA_TYPE, 'preprocessed_test.csv')

    # save files
    print('Saving preprocessed {} dataset into {}... '.format(DATA_TYPE, os.path.join('data', DATA_TYPE)), end='')
    train_label, train_text, test_label, test_text = [], [], [], []
    for data in train_dataset:
        train_label.append(data['label'])
        train_text.append(data['text'])
    for data in test_dataset:
        test_label.append(data['label'])
        test_text.append(data['text'])
    train_data_frame = pd.DataFrame({'label': train_label, 'text': train_text})
    test_data_frame = pd.DataFrame({'label': test_label, 'text': test_text})
    train_data_frame.to_csv(train_file, header=False, index=False)
    test_data_frame.to_csv(test_file, header=False, index=False)
    print('Done.')
