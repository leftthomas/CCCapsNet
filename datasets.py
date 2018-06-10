import os
import sys

from torchnlp.datasets.dataset import Dataset

from utils import GoogleDriveDownloader as gdd
from utils import text_preprocess


def imdb_dataset(directory='data/', data_type='imdb', preprocessing=False, fine_grained=False,
                 verbose=False, text_length=1200, share_id='1nlyc9HOTszLPcwzBtx3vws9b2K18eMxn'):
    """
    Load the IMDB dataset (Large Movie Review Dataset v1.0).

    This is a dataset for binary sentiment classification containing substantially more data than
    previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for training,
    and 25,000 for testing.
    The min length of text about train data is 4, max length of it is 1199; The min length of text
    about test data is 3, max length of it is 930. The average length of text about train data is
    96, the average length of text about test data is 94.

    **Reference:** http://ai.stanford.edu/~amaas/data/sentiment/

    Args:
        directory (str, optional): Directory to cache the dataset.
        data_type (str, optional): Which dataset to use.
        preprocessing (bool, optional): Whether to preprocess the original dataset. If preprocessing
            equals None, it will not download the preprocessed dataset, it will generate preprocessed
            dataset from the original dataset.
        fine_grained (bool, optional): Whether to use fine_grained dataset instead of polarity dataset.
        verbose (bool, optional): Whether to print the dataset details.
        text_length (int, optional): Only load the first text_length words, it only works when
            preprocessing is True.
        share_id (str, optional): Google Drive share ID about the original dataset to download.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset and test
        dataset.

    Example:
        >>> train, test = imdb_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'pos',
          'text': 'movi respect lot memor quot list gem imagin movi joe piscopo funni...'},
         {
          'label': 'pos',
          'text': 'bizarr horror movi fill famou face stolen cristina rain flamingo road...'}]
        >>> test[0:2]
        [{
          'label': 'pos',
          'text': 'movi respect lot memor quot list gem imagin movi joe piscopo funni...'},
         {
          'label': 'pos',
          'text': 'bizarr horror movi fill famou face stolen cristina rain flamingo road...'}]
    """

    if preprocessing:
        gdd.download_file_from_google_drive(share_id, data_type + '_preprocessed.zip', directory + data_type)
        if fine_grained:
            train_file, test_file = 'preprocessed_fine_grained_train.txt', 'preprocessed_fine_grained_test.txt'
        else:
            train_file, test_file = 'preprocessed_train.txt', 'preprocessed_test.txt'
    else:
        gdd.download_file_from_google_drive(share_id, data_type + '_orginal.zip', directory + data_type)
        if fine_grained:
            train_file, test_file = 'orginal_fine_grained_train.txt', 'orginal_fine_grained_test.txt'
        else:
            train_file, test_file = 'orginal_train.txt', 'orginal_test.txt'

    if verbose:
        min_train_length, avg_train_length, max_train_length = sys.maxsize, 0, 0
        min_test_length, avg_test_length, max_test_length = sys.maxsize, 0, 0

    ret = []
    for file_name in [train_file, test_file]:
        with open(os.path.join(directory, data_type, file_name), 'r', encoding='utf-8') as f:
            examples = []
            for line in f.readlines():
                label, text = line.split('\t')
                text = text.rstrip('\n')
                if preprocessing:
                    if len(text.split()) > text_length:
                        text = ' '.join(text.split()[:text_length])
                elif preprocessing is None:
                    text = text_preprocess(text)
                    if len(text.split()) == 0:
                        continue
                if verbose:
                    if file_name == train_file:
                        avg_train_length += len(text.split())
                        if len(text.split()) > max_train_length:
                            max_train_length = len(text.split())
                        if len(text.split()) < min_train_length:
                            min_train_length = len(text.split())
                    if file_name == test_file:
                        avg_test_length += len(text.split())
                        if len(text.split()) > max_test_length:
                            max_test_length = len(text.split())
                        if len(text.split()) < min_test_length:
                            min_test_length = len(text.split())
                examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))

    if verbose:
        print("[!] train length--(min: {}, avg: {}, max: {})".
              format(min_train_length, round(avg_train_length / len(ret[0])), max_train_length))
        print("[!] test length--(min: {}, avg: {}, max: {})".
              format(min_test_length, round(avg_test_length / len(ret[1])), max_test_length))
    return tuple(ret)


def agnews_dataset(directory='data/', preprocessing=False, verbose=False, text_length=1200):
    """
    Load the AG's News Topic Classification dataset (Version 3).

    The AG's news topic classification dataset is constructed by choosing 4 largest classes from
    the original corpus. Each class contains 30,000 training samples and 1,900 testing samples.
    The total number of training samples is 120,000 and testing 7,600.
    The min length of text about train data is 3, max length of it is 91; The min length of text
    about test data is 5, max length of it is 74. The average length of text about train data is
    22, the average length of text about test data is 22.

    **Reference:** http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html

    Example:
        >>> train, test = agnews_dataset()
        >>> train[0:2]
        [{
          'label': 'Business',
          'text': 'wall bear claw back black reuter reuter short seller wall street dwindl band...'},
         {
          'label': 'Business',
          'text': 'carlyl commerci aerospac reuter reuter privat invest firm carlyl group reput...'}]
        >>> test[0:2]
        [{
          'label': 'Business',
          'text': 'wall bear claw back black reuter reuter short seller wall street dwindl band...'},
         {
          'label': 'Business',
          'text': 'carlyl commerci aerospac reuter reuter privat invest firm carlyl group reput...'}]
    """

    return imdb_dataset(directory, 'agnews', train, test,
                        pred_ids=['1plrqZTyhYvSkvKsNaos5hqN6eqjfWMb6', '1dY2ppjVEloLSKAOfnS2oUdai-wR8ISc0'])


def dbpedia_dataset(directory='data/', preprocessing=False, verbose=False, text_length=1200):
    """
    Load the DBPedia Ontology Classification dataset (Version 2).

    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
    from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we
    randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size
    of the training dataset is 560,000 and testing dataset 70,000.
    The min length of text about train data is 1, max length of it is 1001; The min length of text
    about test data is 2, max length of it is 355. The average length of text about train data is 27,
    the average length of text about test data is 27.

    **Reference:** http://dbpedia.org

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
                        pred_ids=['1UVRYZ8B30vepUnfNVjZoqC1srAp_EDfT', '1JPYEPbexNRXq2U05a2dIBFrhjCZdK9Y5'])


def newsgroups_dataset(directory='data/', preprocessing=False, verbose=False, text_length=1200):
    """
    Load the 20 Newsgroups dataset (Version 'bydate').

    The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents,
    partitioned (nearly) evenly across 20 different newsgroups. The total number of training
    samples is 11,293 and testing 7,527.
    The min length of text about train data is 1, max length of it is 6779; The min length of
    text about test data is 1, max length of it is 6142. The average length of text about train
    data is 131, the average length of text about test data is 129.

    **Reference:** http://qwone.com/~jason/20Newsgroups/

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
                        pred_ids=['16uZCEsmwKteEcSCjKaXR-Nw-w0WVwOY7', '1mmiPXs-otrdmh_w5jNjIP6niXVICW1T6'])


def webkb_dataset(directory='data/', preprocessing=False, verbose=False, text_length=1200):
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
    about test data is 1, max length of it is 2082. The average length of text about train data is
    126, the average length of text about test data is 134.

    **Reference:** http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/

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
                        pred_ids=['166VJXbk0WdZIEU527m8LAka7qOv0jfCq', '18dpFqT_-GUOWq6h8KGGAhGDRQCa2_DfP'])


def yahoo_dataset(directory='data/', preprocessing=False, verbose=False, text_length=1200):
    """
    Load the Yahoo! Answers Topic Classification dataset (Version 2).

    The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories.
    Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number
    of training samples is 1,400,000 and testing samples 60,000 in this dataset. After preprocessing,
    the total number of training samples is 1,399,272 and testing samples 59,969 in this dataset.
    We only used the best answer content and the main category information.
    The min length of text about train data is 1, max length of it is 1006; The min length of text
    about test data is 1, max length of it is 491. The average length of text about train data is 37,
    the average length of text about test data is 37.

    **Reference:** https://webscope.sandbox.yahoo.com/catalog.php?datatype=l

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
                        pred_ids=['1TM4AHEJEeb-l6sMRpl2Y5qiaJ4FateJl', '1JstXhPIgzjNOU4Ekeb3ICIOfVVJoto5D'])


def reuters_dataset(directory='data/', preprocessing=False, fine_grained=False, verbose=False, text_length=1200):
    """
    Load the Reuters-21578 R8 or Reuters-21578 R52 dataset (Version 'modApté').

    The Reuters-21578 dataset considers only the documents with a single topic and the classes
    which still have at least one train and one test example, we have 8 of the 10 most frequent
    classes and 52 of the original 90. In total there are 5,485 trainig samples and 2,189 testing
    samples in R8 dataset. The total number of training samples is 6,532 and testing 2,568 in R52
    dataset.
    The min length of text about R8 train data is 4, max length of it is 533; The min length of
    text about R8 test data is 5, max length of it is 484. The min length of text about R52 train
    data is 4, max length of it is 595; The min length of text about R52 test data is 5, max length
    of it is 484. The average length of text about R8 train data is 66, the average length of text
    about R8 test data is 60. The average length of  text about R52 train data is 70, the average
    length of text about R52 test data is 64.

    **Reference:** http://www.daviddlewis.com/resources/testcollections/reuters21578/

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
    return imdb_dataset(directory, 'reuters', train, test, fine_grained, pred_ids=ids)


def yelp_dataset(directory='data/', preprocessing=False, fine_grained=False, verbose=False, text_length=1200):
    """
    Load the Yelp Review Full Star or Yelp Review Polarity dataset (Version 1).

    The Yelp reviews full star dataset is constructed by randomly taking 130,000 training samples and
    10,000 testing samples for each review star from 1 to 5. In total there are 650,000 trainig samples
    and 50,000 testing samples. After preprocessing, in total there are 649,816 trainig samples
    and 49,990 testing samples.
    The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3 and 4
    positive. For each polarity 280,000 training samples and 19,000 testing samples are take randomly.
    In total there are 560,000 trainig samples and 38,000 testing samples. Negative polarity is class 1,
    and positive class 2. After preprocessing, in total there are 559,861 trainig samples and 37,985
    testing samples. Negative polarity is class 1, and positive class 2.
    The min length of text about polarity train data is 1, max length of it is 617; The min length
    of text about polarity test data is 1, max length of it is 504.
    The min length of text about full train data is 1, max length of it is 617; The min length of
    text about full test data is 1, max length of it is 524.
    The average length of text about polarity train data is 53, the average length of text about
    polarity test data is 52.
    The average length of text about full train data is 53, the average length of text about full
    test data is 53.

    **Reference:** http://www.yelp.com/dataset_challenge

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
    return imdb_dataset(directory, 'yelp', train, test, fine_grained, pred_ids=ids)


def amazon_dataset(directory='data/', preprocessing=False, fine_grained=False, verbose=False, text_length=1200):
    """
    Load the Amazon Review Full Score or Amazon Review Polaridy dataset (Version 3).

    The Amazon reviews full score dataset is constructed by randomly taking 600,000 training samples and
    130,000 testing samples for each review score from 1 to 5. In total there are 3,000,000 trainig samples
    and 650,000 testing samples. After preprocessing, in total there are 2,999,979 training samples and
    649,993 testing samples.
    The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and
    5 as positive. For each polarity 1,800,000 training samples and 200,000 testing samples are take randomly.
    In total there are 3,600,000 trainig samples and 400,000 testing samples. Negative polarity is class 1,
    and positive class 2. After preprocessing, in total there are 3,599,981 trainig samples and 399,998 testing
    samples.
    The min length of text about full score train data is 1, max length of it is 141; The min length
    of text about full score test data is 1, max length of it is 132.
    The min length of text about polarity train data is 1, max length of it is 156; The min length
    of text about polarity test data is 1, max length of it is 120.
    The average length of text about polarity train data is 31, the average length of text about
    polarity test data is 31.
    The average length of text about full train data is 32, the average length of text about full
    test data is 32.

    **Reference:** http://jmcauley.ucsd.edu/data/amazon/

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
    return imdb_dataset(directory, 'amazon', train, test, fine_grained, pred_ids=ids)
