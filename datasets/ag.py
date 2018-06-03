import os
import sys

import numpy as np
import pandas as pd
from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract

from .data_utils import text_preprocess


def ag_dataset(directory='data/', train=False, test=False, extracted_name='ag_news', check_files=['ag_news/readme.txt'],
               url='https://drive.google.com/uc?export=download&id=1wNEIR3xNZncHmqzFnTjxX8js5PAOKGnk'):
    """
    Load the AG's News Topic Classification dataset (Version 3).

    The AG's news topic classification dataset is constructed by choosing 4 largest classes
    from the original corpus. Each class contains 30,000 training samples and 1,900 testing
    samples. The total number of training samples is 120,000 and testing 7,600.
    The min length of text about train data is 15, max length of it is 594; The min length
    of text about test data is 42, max length of it is 497.

    **Reference:** http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html

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
        >>> from datasets import ag_dataset
        >>> train = ag_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Business',
          'text': 'wall bear claw back black reuter reuter short seller wall street dwindl band ultra cynic green'},
         {
          'label': 'Business',
          'text': 'carlyl commerci aerospac reuter reuter privat invest firm carlyl group reput make time...'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='ag_news.tar.gz', check_files=check_files)

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
        text_min_length = sys.maxsize
        text_max_length = 0
        for data in csv_file:
            label, title, description = index_to_label[int(data[0]) - 1], data[1], data[2]
            # The title of each document is simply added in the beginning of the document's text.
            if isinstance(title, str) and isinstance(description, str):
                text = text_preprocess(title + ' ' + description)
                if len(text) > text_max_length:
                    text_max_length = len(text)
                if len(text) < text_min_length:
                    text_min_length = len(text)
            else:
                continue
            examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))
        print('text_min_length:' + str(text_min_length))
        print('text_max_length:' + str(text_max_length))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
