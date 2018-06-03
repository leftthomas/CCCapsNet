import os

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def newsgroups_dataset(directory='data/', train=False, test=False, extracted_name='newsgroups',
                       check_files=['newsgroups/20ng-train-stemmed.txt'],
                       url='https://drive.google.com/uc?export=download&id=10NqffTpj_qhyBXaRPr54169O-l1X1pjS'):
    """
    Load the 20 Newsgroups dataset (Version 'bydate').

    The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents,
    partitioned (nearly) evenly across 20 different newsgroups. The total number of training
    samples is 11,293 and testing 7,528.
    The min length of text about train data is 5, max length of it is 42355; The min length
    of text about test data is 1, max length of it is 36809.

    **Reference:** http://qwone.com/~jason/20Newsgroups/

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
        >>> from datasets import newsgroups_dataset
        >>> train = newsgroups_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'alt.atheism',
          'text': 'alt atheism faq atheist resourc archiv name atheism resourc alt...'},
         {
          'label': 'alt.atheism',
          'text': 'alt atheism faq introduct atheism archiv name atheism introduct alt...'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='newsgroups.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in
              [(train, '20ng-train-stemmed.txt'), (test, '20ng-test-stemmed.txt')] if requested]
    for file_name in splits:
        with open(os.path.join(directory, extracted_name, file_name), 'r', encoding='utf-8') as foo:
            examples = []
            for line in foo.readlines():
                label, text = line.split('\t')
                examples.append({'label': label, 'text': text.rstrip('\n')})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
