import os

from google_drive_downloader import GoogleDriveDownloader as gdd
from torchnlp.datasets.dataset import Dataset


def basic_dataset(directory='data/', data_type='imdb', train=False, test=False, fine_grained=False,
                  # train, test
                  share_id=['1nlyc9HOTszLPcwzBtx3vws9b2K18eMxn', '1uSzCdUncgTwIb8cyT_xPoYvxiDN4_xLA']):
    """
    Load the txt dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        data_type (str, optional): Which dataset to use.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        fine_grained (bool, optional): Whether to use fine_grained dataset instead of polarity dataset.
        share_id (str, optional): Google Drive share IDs to download.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset and
        test dataset in order if their respective boolean argument is true.
    """

    if train is False and test is False:
        raise ValueError("The train and test can't all be False.")
    if len(share_id) != 2:
        raise ValueError('The share_id must contains two ids.')
    if fine_grained:
        train_file, test_file = data_type + '_fine_grained_train.txt', data_type + '_fine_grained_test.txt'
    else:
        train_file, test_file = data_type + '_train.txt', data_type + '_test.txt'

    if train:
        gdd.download_file_from_google_drive(file_id=share_id[0], dest_path=directory + train_file)
    if test:
        gdd.download_file_from_google_drive(file_id=share_id[1], dest_path=directory + test_file)

    ret = []
    splits = [file_name for (requested, file_name) in [(train, train_file), (test, test_file)] if requested]
    for file_name in splits:
        with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
            examples = []
            for line in f.readlines():
                label, text = line.split('\t')
                examples.append({'label': label, 'text': text.rstrip('\n')})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def ag_dataset(directory='data/', train=False, test=False):
    """
    Load the AG's News Topic Classification dataset (Version 3).

    The AG's news topic classification dataset is constructed by choosing 4 largest classes
    from the original corpus. Each class contains 30,000 training samples and 1,900 testing
    samples. The total number of training samples is 120,000 and testing 7,600.
    The min length of text about train data is 3, max length of it is 91; The min length
    of text about test data is 5, max length of it is 74.

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

    return basic_dataset(directory, 'agnews', train, test,
                         share_id=['1plrqZTyhYvSkvKsNaos5hqN6eqjfWMb6', '1dY2ppjVEloLSKAOfnS2oUdai-wR8ISc0'])


def dbpedia_dataset(directory='data/', train=False, test=False):
    """
    Load the DBPedia Ontology Classification dataset (Version 2).

    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
    from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we randomly
    choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training
    dataset is 560,000 and testing dataset 70,000.
    The min length of text about train data is 1, max length of it is 1001; The min length
    of text about test data is 2, max length of it is 355.

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

    return basic_dataset(directory, 'dbpedia', train, test,
                         share_id=['1UVRYZ8B30vepUnfNVjZoqC1srAp_EDfT', '1JPYEPbexNRXq2U05a2dIBFrhjCZdK9Y5'])
