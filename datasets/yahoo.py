import csv
import os

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def yahoo_dataset(directory='data/', train=False, test=False, extracted_name='yahoo_answers',
                  check_files=['yahoo_answers/readme.txt'],
                  url='https://link.gimhoy.com/googledrive/aHR0cHM6Ly9kcml2ZS5nb29nbGUuY29tL29wZW4/'
                      'aWQ9MUs2WGpmaTc2Y3R4YjhlZ1lzT0tLRV9HMHduMFV3X1JB.tar.gz'):
    """
    Load the Yahoo! Answers Topic Classification dataset (Version 2).

    The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories.
    Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number
    of training samples is 1,400,000 and testing samples 60,000 in this dataset. From all the answers
    and other meta-information, we only used the best answer content and the main category information.

    **Reference:** https://webscope.sandbox.yahoo.com/catalog.php?datatype=l

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `tar.gz` file.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset and
        test dataset in order if their respective boolean argument is true.

    Example:
        >>> from datasets import yahoo_dataset
        >>> train = yahoo_dataset(train=True)
        >>> train[0:2]
        [{
          'label': '3',
          'title': 'Wall St. Bears Claw Back Into the Black (Reuters)',
          'content': 'Wall St. Bears Claw Back Into the Black (Reuters)',
          'answer': "Reuters - Short-sellers, Wall Street's dwindling..."},
         {
          'label': '3',
          'title': 'Carlyle Looks Toward Commercial Aerospace (Reuters)',
          'content': 'Wall St. Bears Claw Back Into the Black (Reuters)',
          'answer': 'Reuters - Private investment firm Carlyle Group...'}]
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in [(train, 'train.csv'), (test, 'test.csv')] if requested]
    for file_name in splits:
        csv_file = csv.reader(open(os.path.join(directory, extracted_name, file_name), 'r', encoding='utf-8'))
        examples = []
        for data in csv_file:
            examples.append({'label': data[0], 'title': data[1], 'content': data[2], 'answer': data[3]})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
