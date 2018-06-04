import os

from google_drive_downloader import GoogleDriveDownloader as gdd
from torchnlp.datasets.dataset import Dataset


def basic_dataset(directory='data/', data_type='imdb', train=False, test=False, fine_grained=False,
                  # train, test
                  share_id=['1nlyc9HOTszLPcwzBtx3vws9b2K18eMxn', '1uSzCdUncgTwIb8cyT_xPoYvxiDN4_xLA']):
    """
    Load the txt dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        data_type (str, optional): Which dataset to use.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        fine_grained (bool, optional): Whether to use fine_grained dataset instead of polarity dataset.
        share_id (str, optional): Google Drive share IDs to download.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset and
        test dataset in order if their respective boolean argument is true.
    """

    if train is False and test is False:
        raise ValueError("The train and test can't all be False.")
    if len(share_id) != 2:
        raise ValueError('The share_id must contains two ids.')
    if fine_grained:
        train_file, test_file = data_type + '_fine_grained_train.txt', data_type + '_fine_grained_test.txt'
    else:
        train_file, test_file = data_type + '_train.txt', data_type + '_test.txt'

    if train:
        gdd.download_file_from_google_drive(file_id=share_id[0], dest_path=directory + train_file)
    if test:
        gdd.download_file_from_google_drive(file_id=share_id[1], dest_path=directory + test_file)

    ret = []
    splits = [file_name for (requested, file_name) in [(train, train_file), (test, test_file)] if requested]
    for file_name in splits:
        with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
            examples = []
            for line in f.readlines():
                label, text = line.split('\t')
                examples.append({'label': label, 'text': text.rstrip('\n')})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
