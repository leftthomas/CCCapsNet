import os

from google_drive_downloader import GoogleDriveDownloader as gdd
from torchnlp.datasets.dataset import Dataset


def imdb_dataset(directory='data/', data_type='imdb', train=False, test=False, fine_grained=False,
                 # train, test
                 share_id=['1nlyc9HOTszLPcwzBtx3vws9b2K18eMxn', '1uSzCdUncgTwIb8cyT_xPoYvxiDN4_xLA']):
    """
    Load the IMDB dataset (Large Movie Review Dataset v1.0).

    This is a dataset for binary sentiment classification containing substantially more data than
    previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for
    training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text
    and already processed bag of words formats are provided.
    The min length of text about train data is 4, max length of it is 1199; The min length
    of text about test data is 3, max length of it is 930.

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

    Example:
        >>> train = imdb_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'pos',
          'text': 'movi respect lot memor quot list gem imagin movi joe piscopo funni...'},
         {
          'label': 'pos',
          'text': 'bizarr horror movi fill famou face stolen cristina rain flamingo road...'}]
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


def agnews_dataset(directory='data/', train=False, test=False):
    """
    Load the AG's News Topic Classification dataset (Version 3).

    The AG's news topic classification dataset is constructed by choosing 4 largest classes
    from the original corpus. Each class contains 30,000 training samples and 1,900 testing
    samples. The total number of training samples is 120,000 and testing 7,600.
    The min length of text about train data is 3, max length of it is 91; The min length
    of text about test data is 5, max length of it is 74.

    Example:
        >>> train = agnews_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Business',
          'text': 'wall bear claw back black reuter reuter short seller wall street dwindl band ultra cynic green'},
         {
          'label': 'Business',
          'text': 'carlyl commerci aerospac reuter reuter privat invest firm carlyl group reput make time...'}]
    """

    return imdb_dataset(directory, 'agnews', train, test,
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
        >>> train = dbpedia_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Company',
          'text': 'abbott abbott farnham abbott limit british coachbuild busi base farnham surrei...'},
         {
          'label': 'Company',
          'text': 'schwan stabilo schwan stabilo german maker pen write colour cosmet marker highlight...'}]
    """

    return imdb_dataset(directory, 'dbpedia', train, test,
                        share_id=['1UVRYZ8B30vepUnfNVjZoqC1srAp_EDfT', '1JPYEPbexNRXq2U05a2dIBFrhjCZdK9Y5'])


def newsgroups_dataset(directory='data/', train=False, test=False):
    """
    Load the 20 Newsgroups dataset (Version 'bydate').

    The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents,
    partitioned (nearly) evenly across 20 different newsgroups. The total number of training
    samples is 11,293 and testing 7,527.
    The min length of text about train data is 1, max length of it is 6779; The min length
    of text about test data is 1, max length of it is 6142.

    Example:
        >>> train = newsgroups_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'alt.atheism',
          'text': 'alt atheism faq atheist resourc archiv name atheism resourc alt...'},
         {
          'label': 'alt.atheism',
          'text': 'alt atheism faq introduct atheism archiv name atheism introduct alt...'}]
    """

    return imdb_dataset(directory, 'newsgroups', train, test,
                        share_id=['16uZCEsmwKteEcSCjKaXR-Nw-w0WVwOY7', '1mmiPXs-otrdmh_w5jNjIP6niXVICW1T6'])


def webkb_dataset(directory='data/', train=False, test=False):
    """
    Load the World Wide Knowledge Base (Web->Kb) dataset (Version 1).

    The World Wide Knowledge Base (Web->Kb) dataset is collected by the World Wide
    Knowledge Base (Web->Kb) project of the CMU text learning group. These pages
    were collected from computer science departments of various universities in 1997,
    manually classified into seven different classes: student, faculty, staff,
    department, course, project, and other. The classes Department and Staff is
    discarded, because there were only a few pages from each university. The class
    Other is discarded, because pages were very different among this class. The total
    number of training samples is 2,785 and testing 1,383.
    The min length of text about train data is 1, max length of it is 20628; The min length
    of text about test data is 1, max length of it is 2082.

    Example:
        >>> train = webkb_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'student',
          'text': 'brian comput scienc depart univers wisconsin dayton street madison offic...'}
         {
          'label': 'student',
          'text': 'denni swanson web page mail pop uki offic hour comput lab offic anderson...'}]
    """

    return imdb_dataset(directory, 'webkb', train, test,
                        share_id=['166VJXbk0WdZIEU527m8LAka7qOv0jfCq', '18dpFqT_-GUOWq6h8KGGAhGDRQCa2_DfP'])
