import os

import numpy as np
import pandas as pd
from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract

from .data_utils import text_preprocess


def yahoo_dataset(directory='data/', train=False, test=False, extracted_name='yahoo_answers',
                  check_files=['yahoo_answers/readme.txt'],
                  url='https://drive.google.com/uc?export=download&id=1K6Xjfi76ctxb8egYsOKKE_G0wn0Uw_RA'):
    """
    Load the Yahoo! Answers Topic Classification dataset (Version 2).

    The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories.
    Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number
    of training samples is 1,400,000 and testing samples 60,000 in this dataset. From all the answers
    and other meta-information, we only used the best answer content and the main category information.
    The min length of text about train data is 15, max length of it is 594; The min length
    of text about test data is 42, max length of it is 497.

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
          'label': 'Computers & Internet',
          'text': "why doesn't an optical mouse work on a glass table?"},
         {
          'label': 'Sports',
          'text': 'What is the best off-road motorcycle trail ?'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='yahoo_answers.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in [(train, 'train.csv'), (test, 'test.csv')] if requested]
    index_to_label = []
    with open(os.path.join(directory, extracted_name, 'classes.txt'), 'r', encoding='utf-8') as foo:
        for line in foo.readlines():
            line = line.rstrip('\n')
            index_to_label.append(line)
    for file_name in splits:
        csv_file = np.array(pd.read_csv(os.path.join(directory, extracted_name, file_name), header=None)).tolist()
        examples = []
        for data in csv_file:
            label, title, content, answer = index_to_label[int(data[0]) - 1], data[1], data[2], data[3]
            # The title of each document is simply added in the beginning of the document's text.
            if isinstance(title, str) and isinstance(content, str) and isinstance(answer, str):
                text = text_preprocess(title + ' ' + content + ' ' + answer)
            else:
                continue
            examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
