import os
import sys

import numpy as np
import pandas as pd
from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract

from .data_utils import text_preprocess


def dbpedia_dataset(directory='data/', train=False, test=False, extracted_name='dbpedia',
                    check_files=['dbpedia/readme.txt'],
                    url='https://drive.google.com/uc?export=download&id=13WPe20BmrI_a9vgRa91PxfTrbBcR-1Ee'):
    """
    Load the DBPedia Ontology Classification dataset (Version 2).

    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
    from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we randomly
    choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training
    dataset is 560,000 and testing dataset 70,000.
    The min length of text about train data is 1, max length of it is 1001; The min length
    of text about test data is 2, max length of it is 355.

    **Reference:** http://dbpedia.org

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
        >>> from datasets import dbpedia_dataset
        >>> train = dbpedia_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Company',
          'text': 'abbott abbott farnham abbott limit british coachbuild busi base farnham surrei...'},
         {
          'label': 'Company',
          'text': 'schwan stabilo schwan stabilo german maker pen write colour cosmet marker highlight...'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='dbpedia.tar.gz', check_files=check_files)

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
                if len(text.split()) == 0:
                    continue
                else:
                    if len(text.split()) > text_max_length:
                        text_max_length = len(text.split())
                    if len(text.split()) < text_min_length:
                        text_min_length = len(text.split())
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
