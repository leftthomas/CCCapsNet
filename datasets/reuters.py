import os
import sys

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def reuters_dataset(directory='data/', train=False, test=False, extracted_name='reuters',
                    check_files=['reuters/r8-train-stemmed.txt'],
                    url='https://drive.google.com/uc?export=download&id=1grhm1NvEty46XbLBSQsFqEQmvzyn2kX0',
                    fine_grained=False):
    """
    Load the Reuters-21578 R8 or Reuters-21578 R52 dataset (Version 'modApté').

    The Reuters-21578 dataset considers only the documents with a single topic and the
    classes which still have at least one train and one test example, we have 8 of the 10
    most frequent classes and 52 of the original 90. In total there are 5,485 trainig samples
    and 2,189 testing samples in R8 dataset. The total number of training samples is 6,532
    and testing 2,568 in R52 dataset.
    The min length of text about R8 train data is 4, max length of it is 533; The min length
    of text about R8 test data is 5, max length of it is 484.
    The min length of text about R52 train data is 4, max length of it is 595; The min length
    of text about R52 test data is 5, max length of it is 484.

    **Reference:** http://www.daviddlewis.com/resources/testcollections/reuters21578/

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `tar.gz` file.
        fine_grained (bool, optional): Whether to use 52-class instead of 8-class labeling. Which
        means using R52 dataset instead of R8 dataset

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset and
        test dataset in order if their respective boolean argument is true.

    Example:
        >>> from datasets import reuters_dataset
        >>> train = reuters_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'earn',
          'text': 'champion product approv stock split champion product inc board director...'}
         {
          'label': 'acq',
          'text': 'comput termin system cpml complet sale comput termin system inc complet...'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='reuters.tar.gz', check_files=check_files)

    if fine_grained:
        train_file_name, test_file_name = 'r52-train-stemmed.txt', 'r52-test-stemmed.txt'
    else:
        train_file_name, test_file_name = 'r8-train-stemmed.txt', 'r8-test-stemmed.txt'
    ret = []
    splits = [file_name for (requested, file_name) in [(train, train_file_name), (test, test_file_name)] if requested]
    for file_name in splits:
        with open(os.path.join(directory, extracted_name, file_name), 'r', encoding='utf-8') as foo:
            examples = []
            text_min_length = sys.maxsize
            text_max_length = 0
            for line in foo.readlines():
                label, text = line.split('\t')
                if len(text.split()) == 0:
                    continue
                else:
                    if len(text.split()) > text_max_length:
                        text_max_length = len(text.split())
                    if len(text.split()) < text_min_length:
                        text_min_length = len(text.split())
                examples.append({'label': label, 'text': text.rstrip('\n')})
        ret.append(Dataset(examples))
        print('text_min_length:' + str(text_min_length))
        print('text_max_length:' + str(text_max_length))

    train_file = 'data/reuters_fine_grained_train.txt'
    train_f = open(train_file, 'w')
    for train_data in ret[0]:
        train_f.write(train_data['label'] + '\t' + train_data['text'] + '\n')
    test_file = 'data/reuters_fine_grained_test.txt'
    test_f = open(test_file, 'w')
    for test_data in ret[1]:
        test_f.write(test_data['label'] + '\t' + test_data['text'] + '\n')

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
