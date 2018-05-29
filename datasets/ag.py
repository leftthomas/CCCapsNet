import glob
import os

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def ag_dataset(directory='data/', train=False, test=False, check_files=['ag_news/readme.txt'],
               url='https://link.gimhoy.com/googledrive/aHR0cHM6Ly9kcml2ZS5nb29nbGUuY29tL29wZW4/'
                   'aWQ9MXdORUlSM3hOWm5jSG1xekZuVGp4WDhqczVQQU9LR25r.tar.gz'):
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
          'text': 'For a movie that gets no respect there sure are a lot of memorable quotes...',
          'sentiment': 'pos'
        }, {
          'text': 'Bizarre horror movie filled with famous faces but stolen by Cristina Raines...',
          'sentiment': 'pos'
        }]
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    for _ in range(train + test):
        examples = []
        for sentiment in sentiments:
            for filename in glob.iglob(os.path.join(directory, sentiment, '*.txt')):
                with open(filename, 'r', encoding="utf-8") as f:
                    text = f.readline()
                examples.append({
                    'text': text,
                    'sentiment': sentiment,
                })
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
