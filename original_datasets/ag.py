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
            if (not isinstance(title, str)) and (not isinstance(description, str)):
                continue
            else:
                if isinstance(title, str) and (not isinstance(description, str)):
                    text = title
                elif isinstance(description, str) and (not isinstance(title, str)):
                    text = description
                else:
                    # The title of each document is simply added in the beginning of the document's text.
                    text = title + ' ' + description
                text = text_preprocess(text)
                if len(text.split()) == 0:
                    continue
                else:
                    if len(text.split()) > text_max_length:
                        text_max_length = len(text.split())
                    if len(text.split()) < text_min_length:
                        text_min_length = len(text.split())

            examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))
        print('text_min_length:' + str(text_min_length))
        print('text_max_length:' + str(text_max_length))

    train_file = 'data/agnews_train.txt'
    train_f = open(train_file, 'w')
    for train_data in ret[0]:
        train_f.write(train_data['label'] + '\t' + train_data['text'] + '\n')
    train_f.close()
    test_file = 'data/agnews_test.txt'
    test_f = open(test_file, 'w')
    for test_data in ret[1]:
        test_f.write(test_data['label'] + '\t' + test_data['text'] + '\n')
    test_f.close()

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
