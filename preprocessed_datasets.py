import os

from google_drive_downloader import GoogleDriveDownloader as gdd
from torchnlp.datasets.dataset import Dataset


def imdb_dataset(directory='data/', data_type='imdb', train=False, test=False, fine_grained=False,
                 # train, test
                 share_id=['1nlyc9HOTszLPcwzBtx3vws9b2K18eMxn', '1uSzCdUncgTwIb8cyT_xPoYvxiDN4_xLA']):
    """
    Load the IMDB dataset (Large Movie Review Dataset v1.0).

    This is a dataset for binary sentiment classification containing substantially more data than
    previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for training,
    and 25,000 for testing.
    The min length of text about train data is 4, max length of it is 1199; The min length of text
    about test data is 3, max length of it is 930.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about train data is 96, the average length of text about test data is 94.

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
        avg_train_length = 0
    if test:
        gdd.download_file_from_google_drive(file_id=share_id[1], dest_path=directory + test_file)
        avg_test_length = 0

    ret = []
    splits = [file_name for (requested, file_name) in [(train, train_file), (test, test_file)] if requested]
    for file_name in splits:
        with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
            examples = []
            for line in f.readlines():
                label, text = line.split('\t')
                # we only need the first 1200 words
                if len(text.split()) > 1200:
                    text = ' '.join(text.split()[:1200])
                if file_name == train_file:
                    avg_train_length += len(text.split())
                if file_name == test_file:
                    avg_test_length += len(text.split())
                examples.append({'label': label, 'text': text.rstrip('\n')})
        ret.append(Dataset(examples))

    if train:
        print("[!] avg_train_length: {}".format(round(avg_train_length / len(ret[0]))))
    if test:
        if train:
            print("[!] avg_test_length: {}".format(round(avg_test_length / len(ret[1]))))
        else:
            print("[!] avg_test_length: {}".format(round(avg_test_length / len(ret[0]))))
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def agnews_dataset(directory='data/', train=False, test=False):
    """
    Load the AG's News Topic Classification dataset (Version 3).

    The AG's news topic classification dataset is constructed by choosing 4 largest classes from
    the original corpus. Each class contains 30,000 training samples and 1,900 testing samples.
    The total number of training samples is 120,000 and testing 7,600.
    The min length of text about train data is 3, max length of it is 91; The min length of text
    about test data is 5, max length of it is 74.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about train data is 22, the average length of text about test data is 22.

    Example:
        >>> train = agnews_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Business',
          'text': 'wall bear claw back black reuter reuter short seller wall street dwindl band...'},
         {
          'label': 'Business',
          'text': 'carlyl commerci aerospac reuter reuter privat invest firm carlyl group reput...'}]
    """

    return imdb_dataset(directory, 'agnews', train, test,
                        share_id=['1plrqZTyhYvSkvKsNaos5hqN6eqjfWMb6', '1dY2ppjVEloLSKAOfnS2oUdai-wR8ISc0'])


def dbpedia_dataset(directory='data/', train=False, test=False):
    """
    Load the DBPedia Ontology Classification dataset (Version 2).

    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
    from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we
    randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size
    of the training dataset is 560,000 and testing dataset 70,000.
    The min length of text about train data is 1, max length of it is 1001; The min length of text
    about test data is 2, max length of it is 355.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about train data is 27, the average length of text about test data is 27.

    Example:
        >>> train = dbpedia_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Company',
          'text': 'abbott abbott farnham abbott limit british coachbuild busi base farnham...'},
         {
          'label': 'Company',
          'text': 'schwan stabilo schwan stabilo german maker pen write colour cosmet marker...'}]
    """

    return imdb_dataset(directory, 'dbpedia', train, test,
                        share_id=['1UVRYZ8B30vepUnfNVjZoqC1srAp_EDfT', '1JPYEPbexNRXq2U05a2dIBFrhjCZdK9Y5'])


def newsgroups_dataset(directory='data/', train=False, test=False):
    """
    Load the 20 Newsgroups dataset (Version 'bydate').

    The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents,
    partitioned (nearly) evenly across 20 different newsgroups. The total number of training
    samples is 11,293 and testing 7,527.
    The min length of text about train data is 1, max length of it is 6779; The min length of text
    about test data is 1, max length of it is 6142.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about train data is 131, the average length of text about test data is 129.

    Example:
        >>> train = newsgroups_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'alt.atheism',
          'text': 'alt atheism faq atheist resourc archiv name atheism resourc alt...'},
         {
          'label': 'alt.atheism',
          'text': 'alt atheism faq introduct atheism archiv name atheism introduct...'}]
    """

    return imdb_dataset(directory, 'newsgroups', train, test,
                        share_id=['16uZCEsmwKteEcSCjKaXR-Nw-w0WVwOY7', '1mmiPXs-otrdmh_w5jNjIP6niXVICW1T6'])


def webkb_dataset(directory='data/', train=False, test=False):
    """
    Load the World Wide Knowledge Base (Web->Kb) dataset (Version 1).

    The World Wide Knowledge Base (Web->Kb) dataset is collected by the World Wide Knowledge Base
    (Web->Kb) project of the CMU text learning group. These pages were collected from computer
    science departments of various universities in 1997, manually classified into seven different
    classes: student, faculty, staff, department, course, project, and other. The classes Department
    and Staff is discarded, because there were only a few pages from each university. The class Other
    is discarded, because pages were very different among this class. The total number of training
    samples is 2,785 and testing 1,383.
    The min length of text about train data is 1, max length of it is 20628; The min length of text
    about test data is 1, max length of it is 2082.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about train data is 126, the average length of text about test data is 134.

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


def yahoo_dataset(directory='data/', train=False, test=False):
    """
    Load the Yahoo! Answers Topic Classification dataset (Version 2).

    The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories.
    The total number of training samples is 1,399,272 and testing samples 59,969 in this dataset.
    We only used the best answer content and the main category information.
    The min length of text about train data is 1, max length of it is 1006; The min length of text
    about test data is 1, max length of it is 491.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about train data is 37, the average length of text about test data is 37.

    Example:
        >>> train = yahoo_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'Computers & Internet',
          'text': 'doesn optic mous work glass tabl surfac optic mice led camera rapidli...'},
         {
          'label': 'Sports',
          'text': 'road motorcycl trail long distanc trail hear mojav road amaz nsearch onlin'}]
    """

    return imdb_dataset(directory, 'yahoo', train, test,
                        share_id=['1TM4AHEJEeb-l6sMRpl2Y5qiaJ4FateJl', '1JstXhPIgzjNOU4Ekeb3ICIOfVVJoto5D'])


def reuters_dataset(directory='data/', train=False, test=False, fine_grained=False):
    """
    Load the Reuters-21578 R8 or Reuters-21578 R52 dataset (Version 'modApté').

    The Reuters-21578 dataset considers only the documents with a single topic and the classes
    which still have at least one train and one test example, we have 8 of the 10 most frequent
    classes and 52 of the original 90. In total there are 5,485 trainig samples and 2,189 testing
    samples in R8 dataset. The total number of training samples is 6,532 and testing 2,568 in R52
    dataset.
    The min length of text about R8 train data is 4, max length of it is 533; The min length of
    text about R8 test data is 5, max length of it is 484.
    The min length of text about R52 train data is 4, max length of it is 595; The min length of
    text about R52 test data is 5, max length of it is 484.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about R8 train data is 66, the average length of text about R8
    test data is 60.
    The average length of text about R52 train data is 70, the average length of text about R52
    test data is 64.

    Example:
        >>> train = reuters_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'earn',
          'text': 'champion product approv stock split champion product inc board director...'}
         {
          'label': 'acq',
          'text': 'comput termin system cpml complet sale comput termin system inc complet...'}]
    """

    if fine_grained:
        ids = ['1JL83q8YoyaffLxSJsrJSAGYSDVcZotNR', '1qQqBUQTGTVwotRPuDwfpHry5Qw77-Dix']
    else:
        ids = ['1jL06ZqR74fKYsMwAFwyGXE_KmZ_b1yGZ', '1QDgNKHyaCTwEdjeN2XJE1nUwKmloPkU9']
    return imdb_dataset(directory, 'reuters', train, test, fine_grained, share_id=ids)


def yelp_dataset(directory='data/', train=False, test=False, fine_grained=False):
    """
    Load the Yelp Review Full Star or Yelp Review Polarity dataset (Version 1).

    The Yelp reviews full star dataset is constructed by randomly taking training samples and
    testing samples for each review star from 1 to 5. In total there are 649,816 trainig samples
    and 49,990 testing samples.
    The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3
    and 4 positive. In total there are 559,861 trainig samples and 37,985 testing samples. Negative
    polarity is class 1, and positive class 2.
    The min length of text about polarity train data is 1, max length of it is 617; The min length
    of text about polarity test data is 1, max length of it is 504.
    The min length of text about full train data is 1, max length of it is 617; The min length of
    text about full test data is 1, max length of it is 524.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about polarity train data is 53, the average length of text about
    polarity test data is 52.
    The average length of text about full train data is 53, the average length of text about full
    test data is 53.

    Example:
        >>> train = yelp_dataset(train=True)
        >>> train[0:2]
        [{
          'label': '1',
          'text': "frustrat goldberg patient repeat experi doctor nyc good doctor terribl staff..."}
         {
          'label': '2',
          'text': "goldberg year patient start mhmg great year big pictur gyn markoff found..."}]
    """

    if fine_grained:
        ids = ['1hRGkaOnYNtjhRXIm43Oo643GiStrxvJ0', '1D8EckH1KPfrfsIV3nIU2eOsFtTGTG7DP']
    else:
        ids = ['1twA4DhJ2mnWh2aQr0qK1UlgqKSmcxuZp', '1KFt3vAZVyUkAnrIvKG7chYXkH0ez49ph']
    return imdb_dataset(directory, 'yelp', train, test, fine_grained, share_id=ids)


def amazon_dataset(directory='data/', train=False, test=False, fine_grained=False):
    """
    Load the Amazon Review Full Score or Amazon Review Polaridy dataset (Version 3).

    The Amazon reviews full score dataset is constructed by randomly taking training samples and
    testing samples for each review score from 1 to 5. In total there are 2,999,979 training samples
    and 649,993 testing samples.
    The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative,
    and 4 and 5 as positive. In total there are 3,599,981 trainig samples and 399,998 testing samples.
    Negative polarity is class 1, and positive class 2.
    The min length of text about full score train data is 1, max length of it is 141; The min length
    of text about full score test data is 1, max length of it is 132.
    The min length of text about polarity train data is 1, max length of it is 156; The min length
    of text about polarity test data is 1, max length of it is 120.
    -------------------------------------Processing Step 2---------------------------------------
    The average length of text about polarity train data is 31, the average length of text about
    polarity test data is 31.
    The average length of text about full train data is 32, the average length of text about full
    test data is 32.

    Example:
        >>> train = amazon_dataset(train=True)
        >>> train[0:2]
        [{
          'label': '2',
          'text': 'stune gamer sound track beauti paint seneri mind recomend peopl hate vid...'}
         {
          'label': '2',
          'text': 'soundtrack read lot review game soundtrack figur write review disagre bit...'}]
    """

    if fine_grained:
        ids = ['1IegvAdxzTye3XLybtfUD1UgNtDzVXn3y', '1fHeXimRtpi2M1EMsZ6phT2QZ-696gofm']
    else:
        ids = ['1Wxahg6ipC9OFnzGH901S6NIVYNILS0ND', '1dbAsmrtGxk9qIVE5DCNxIidhKHMz6PaO']
    return imdb_dataset(directory, 'amazon', train, test, fine_grained, share_id=ids)
