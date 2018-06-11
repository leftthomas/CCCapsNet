import argparse
import os
import re
import warnings
import zipfile
from os import makedirs
from os.path import dirname, exists
from porterstemmer import Stemmer
from sys import stdout

import requests
import torch
import torch.nn.functional as F
from torch import nn
from torchnlp.text_encoders import WhitespaceEncoder, IdentityEncoder
from torchnlp.text_encoders.reserved_tokens import PADDING_TOKEN, UNKNOWN_TOKEN
from torchnlp.utils import datasets_iterator
from torchnlp.utils import pad_batch

from datasets import imdb_dataset, agnews_dataset, amazon_dataset, dbpedia_dataset, newsgroups_dataset, \
    reuters_dataset, webkb_dataset, yahoo_dataset, yelp_dataset, cade_dataset, sogou_dataset


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, file_name, dest_path, overwrite=False):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.

        Args:
            file_id (str): the file identifier. You can obtain it from the sherable link.
            file_name (str): the file name.
            dest_path (str): the destination where to save the downloaded file.
            overwrite (bool): optional, if True forces re-download and overwrite.

        Returns:
            None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(os.path.join(dest_path, file_name)) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_name, dest_path), end='')
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            GoogleDriveDownloader._save_response_content(response, os.path.join(dest_path, file_name))
            print('Done.')

            try:
                print('Unzipping...', end='')
                stdout.flush()
                with zipfile.ZipFile(os.path.join(dest_path, file_name), 'r') as z:
                    z.extractall(destination_directory)
                print('Done.')
            except zipfile.BadZipfile:
                warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_name))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()


stopwords = []
with open(os.path.join('data', 'stopwords.txt'), 'r', encoding='utf-8') as foo:
    for line in foo.readlines():
        line = line.rstrip('\n')
        stopwords.append(line)


def text_preprocess(text):
    # Substitute TAB, NEWLINE and RETURN characters by SPACE.
    text = re.sub('[\t\n\r]', ' ', text)
    # Keep only letters (that is, turn punctuation, numbers, etc. into SPACES).
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Turn all letters to lowercase.
    text = text.lower()
    # Substitute multiple SPACES by a single SPACE.
    text = ' '.join(text.split())
    # Remove words that are less than 3 characters long. For example, removing "he" but keeping "him"
    text = ' '.join(word for word in text.split() if len(word) >= 3)
    # Remove the 524 SMART stopwords (the original stop word list contains 571 words, but there are 47 words contain
    # hyphens, so we removed them, and we found the word 'would' appears twice, so we also removed it, the final stop
    # word list contains 523 words). Some of them had already been removed, because they were shorter than 3 characters.
    # the original stop word list can be found from http://www.lextek.com/manuals/onix/stopwords2.html.
    text = ' '.join(word for word in text.split() if word not in stopwords)
    # Apply Porter's Stemmer to the remaining words.
    stemmer = Stemmer()
    text = ' '.join(stemmer(word) for word in text.split())
    # Substitute multiple SPACES by a single SPACE.
    text = ' '.join(text.split())
    return text


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

    if FINE_GRAINED:
        train_file = os.path.join('data', DATA_TYPE, 'preprocessed_fine_grained_train.txt')
        test_file = os.path.join('data', DATA_TYPE, 'preprocessed_fine_grained_test.txt')
    else:
        train_file = os.path.join('data', DATA_TYPE, 'preprocessed_train.txt')
        test_file = os.path.join('data', DATA_TYPE, 'preprocessed_train.txt')

    # save files
    print('Saving preprocessed {} dataset into {}... '.format(DATA_TYPE, os.path.join('data', DATA_TYPE)), end='')
    train_f = open(train_file, 'w')
    for data in train_dataset:
        train_f.write(data['label'] + '\t' + data['text'] + '\n')
    train_f.close()

    test_f = open(test_file, 'w')
    for data in test_dataset:
        test_f.write(data['label'] + '\t' + data['text'] + '\n')
    test_f.close()
    print('Done.')
