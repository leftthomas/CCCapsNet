import os

import numpy as np
import pandas as pd
from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract

from .data_utils import text_preprocess


def sogou_dataset(directory='data/', train=False, test=False, extracted_name='sogou_news',
                  check_files=['sogou_news/readme.txt'],
                  url='https://drive.google.com/uc?export=download&id=1D7JkMRSUNvNVYK98I0OETN94LjNtuLlx'):
    """
    Load the Sogou News Topic Classification dataset (Version 3).

    The Sogou news topic classification dataset is constructed by manually labeling each news article
    according to its URL, which represents roughly the categorization of news in their websites. We
    chose 5 largest categories for the dataset, each having 90,000 samples for training and 12,000 for
    testing. The Pinyin texts are converted using pypinyin combined with jieba Chinese segmentation system.
    In total there are 450,000 training samples and 60,000 testing samples.

    **Reference:** http://www.sogou.com/labs/dl/ca.html and http://www.sogou.com/labs/dl/cs.html

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
        >>> from datasets import sogou_dataset
        >>> train = sogou_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'automobile',
          'title': '2008 di4 qi1 jie4 qi1ng da3o guo2 ji4 che1 zha3n me3i nv3 mo2 te4 ',
          'content': '2008di4 qi1 jie4 qi1ng da3o guo2 ji4 che1 zha3n yu2...'}
         {
          'label': 'automobile',
          'title': ' zho1ng hua2 ju4n jie2 FRV ya4o shi ',
          'content': 'tu2 we2i zho1ng hua2 ju4n jie2 FRV ya4o shi .'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='sogou_news.tar.gz', check_files=check_files)

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
            label, title, description = index_to_label[int(data[0]) - 1], data[1], data[2]
            # The title of each document is simply added in the beginning of the document's text.
            text = text_preprocess(title + ' ' + description)
            examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
