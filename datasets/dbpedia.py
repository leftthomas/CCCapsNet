import csv
import os

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def dbpedia_dataset(directory='data/', train=False, test=False, extracted_name='dbpedia',
                    check_files=['dbpedia/readme.txt'],
                    url='https://link.gimhoy.com/googledrive/aHR0cHM6Ly9kcml2ZS5nb29nbGUuY29tL29wZW4/'
                        'aWQ9MTNXUGUyMEJtcklfYTl2Z1JhOTFQeGZUcmJCY1ItMUVl.tar.gz'):
    """
    Load the DBPedia Ontology Classification dataset (Version 2).

    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
    from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we randomly
    choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training
    dataset is 560,000 and testing dataset 70,000.

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
          'label': '1',
          'title': 'E. D. Abbott Ltd',
          'content': ' Abbott of Farnham E D Abbott Limited was a British coachbuilding ...'},
         {
          'label': '1',
          'title': 'Schwan-Stabilo',
          'content': " Schwan-STABILO is a German maker of pens for writing colouring ..."}]
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in [(train, 'train.csv'), (test, 'test.csv')] if requested]
    for file_name in splits:
        csv_file = csv.reader(open(os.path.join(directory, extracted_name, file_name), 'r', encoding='utf-8'))
        examples = []
        for data in csv_file:
            examples.append({'label': data[0], 'title': data[1], 'content': data[2]})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
