import warnings
import zipfile
from os import makedirs
from os.path import dirname, exists
from sys import stdout

import requests
import torch
import torch.nn.functional as F
from torch import nn
from torchnlp.text_encoders import WhitespaceEncoder, IdentityEncoder
from torchnlp.text_encoders.reserved_tokens import PADDING_TOKEN, UNKNOWN_TOKEN
from torchnlp.utils import datasets_iterator
from torchnlp.utils import pad_batch

from preprocessed_datasets import imdb_dataset, agnews_dataset, amazon_dataset, dbpedia_dataset, newsgroups_dataset, \
    reuters_dataset, webkb_dataset, yahoo_dataset, yelp_dataset


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, file_name, dest_path, overwrite=False, unzip=False):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.

        Args:
            file_id (str): the file identifier. You can obtain it from the sherable link.
            file_name (str): the file name.
            dest_path (str): the destination where to save the downloaded file.
            overwrite (bool): optional, if True forces re-download and overwrite.
            unzip (bool): optional, if True unzips a file. If the file is not a zip file, ignores it.

        Returns:
            None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_name, dest_path), end='')
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            GoogleDriveDownloader._save_response_content(response, dest_path)
            print('Done.')

            if unzip:
                try:
                    print('Unzipping...', end='')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

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
    label_corpus = [row['label'] for row in datasets_iterator(dataset[0], )]
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
