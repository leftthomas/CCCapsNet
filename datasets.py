import os
import sys

import numpy as np
import pandas as pd
from torchnlp.datasets.dataset import Dataset

from data_utils import GoogleDriveDownloader as gdd
from data_utils import text_preprocess


def imdb_dataset(directory='data/', data_type='imdb', preprocessing=False, fine_grained=False,
                 verbose=False, text_length=2700, share_id='1IuhDyB9D0PpSrmRjjcmSPpQbcb22HVTe'):
    """
    Load the IMDB dataset (Large Movie Review Dataset v1.0).

    This is a dataset for binary sentiment classification containing substantially more data than
    previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for training,
    and 25,000 for testing.
    After preprocessing, the total number of training samples is 25,000 and testing samples 25,000.
    The min length of text about train data is 11, max length is 2,803, average length is 281; the
    min length of text about test data is 8, max length is 2,709, average length is 275.

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
          'text': 'for a movie that gets no respect there sure are a lot of memorable quotes...'},
         {
          'label': 'pos',
          'text': 'bizarre horror movie filled with famous faces but stolen by cristina raines...'}]
        >>> test[0:2]
        [{
          'label': 'pos',
          'text': 'based on an actual story , john boorman shows the struggle of an american...'},
         {
          'label': 'pos',
          'text': 'this is a gem as a film four production the anticipated quality was indeed...'}]
    """

    # other dataset have been set before, only imdb should be set here
    if preprocessing and data_type == 'imdb':
        share_id = '1naVVErkRQNNJXTA6X_X6YrJY0jPOeuPh'

    if preprocessing:
        gdd.download_file_from_google_drive(share_id, data_type + '_preprocessed.zip', directory + data_type)
        if fine_grained:
            train_file, test_file = 'preprocessed_fine_grained_train.csv', 'preprocessed_fine_grained_test.csv'
        else:
            train_file, test_file = 'preprocessed_train.csv', 'preprocessed_test.csv'
    else:
        gdd.download_file_from_google_drive(share_id, data_type + '_orginal.zip', directory + data_type)
        if fine_grained:
            train_file, test_file = 'orginal_fine_grained_train.csv', 'orginal_fine_grained_test.csv'
        else:
            train_file, test_file = 'orginal_train.csv', 'orginal_test.csv'

    if verbose:
        min_train_length, avg_train_length, max_train_length = sys.maxsize, 0, 0
        min_test_length, avg_test_length, max_test_length = sys.maxsize, 0, 0

    ret = []
    for file_name in [train_file, test_file]:
        csv_file = np.array(pd.read_csv(os.path.join(directory, data_type, file_name), header=None)).tolist()
        examples = []
        for label, text in csv_file:
            label, text = str(label), str(text)
            if preprocessing:
                if len(text.split()) > text_length:
                    text = ' '.join(text.split()[:text_length])
            elif preprocessing is None:
                text = text_preprocess(text, data_type)
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
        print('[!] train samples: {} length--(min: {}, avg: {}, max: {})'.
              format(len(ret[0]), min_train_length, round(avg_train_length / len(ret[0])), max_train_length))
        print('[!] test samples: {} length--(min: {}, avg: {}, max: {})'.
              format(len(ret[1]), min_test_length, round(avg_test_length / len(ret[1])), max_test_length))
    return tuple(ret)


def newsgroups_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the 20 Newsgroups dataset (Version 'bydate').

    The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents,
    partitioned (nearly) evenly across 20 different newsgroups. The total number of training
    samples is 11,293 and testing 7,527.
    After preprocessing, the total number of training samples is 11,293 and testing samples 7,527.
    The min length of text about train data is 1, max length is 6,779, average length is 143; the
    min length of text about test data is 1, max length is 6,142, average length is 139.

    **Reference:** http://qwone.com/~jason/20Newsgroups/

    Example:
        >>> train, test = newsgroups_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'alt.atheism',
          'text': 'alt atheism faq atheist resourc archiv name atheism resourc alt atheism...'},
         {
          'label': 'alt.atheism',
          'text': 'alt atheism faq introduct atheism archiv name atheism introduct alt...'}]
        >>> test[0:2]
        [{
          'label': 'alt.atheism',
          'text': 'bibl quiz answer articl healta saturn wwc edu healta saturn wwc edu...'},
         {
          'label': 'alt.atheism',
          'text': 'amus atheist and agnost articl timmbak mcl timmbak mcl ucsb edu clam bake...'}]
    """

    share_id = '1y8M5yf0DD21ox3K76xJyoCkGIU1Zc4iq' if preprocessing else '1tk8vB1RsptrFg-yLLxWAZazFLSoQOw7T'
    return imdb_dataset(directory, 'newsgroups', preprocessing, verbose=verbose, text_length=text_length,
                        share_id=share_id)


def reuters_dataset(directory='data/', preprocessing=False, fine_grained=False, verbose=False, text_length=2700):
    """
    Load the Reuters-21578 R8 or Reuters-21578 R52 dataset (Version 'modApté').

    The Reuters-21578 dataset considers only the documents with a single topic and the classes
    which still have at least one train and one test example, we have 8 of the 10 most frequent
    classes and 52 of the original 90. In total there are 5,485 trainig samples and 2,189 testing
    samples in R8 dataset. The total number of training samples is 6,532 and testing 2,568 in R52
    dataset.
    After preprocessing, the total number of training samples is 5,485 and testing samples 2,189.
    The min length of text about train data is 4, max length is 533, average length is 66; the min
    length of text about test data is 5, max length is 484, average length is 60. (R8)
    After preprocessing, the total number of training samples is 6,532 and testing samples 2,568.
    The min length of text about train data is 4, max length is 595, average length is 70; the min
    length of text about test data is 5, max length is 484, average length is 64. (R52)

    **Reference:** http://www.daviddlewis.com/resources/testcollections/reuters21578/

    Example:
        >>> train, test = reuters_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'earn',
          'text': 'champion product approv stock split champion product inc board director...'}
         {
          'label': 'acq',
          'text': 'comput termin system cpml complet sale comput termin system inc complet...'}]
        >>> test[0:2]
        [{
          'label': 'trade',
          'text': 'asian export fear damag japan rift mount trade friction and japan rais...'},
         {
          'label': 'grain',
          'text': 'china daili vermin eat pct grain stock survei provinc and citi show...'}]
    """

    share_id = '1CY3W31rdagEJ8Kr5gHPeRgS1GVks-YVv' if preprocessing else '18U0eEO31OlgG6IHIRO5D582yFqzG7JWX'
    return imdb_dataset(directory, 'reuters', preprocessing, fine_grained, verbose, text_length, share_id)


def webkb_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the World Wide Knowledge Base (Web->Kb) dataset (Version 1).

    The World Wide Knowledge Base (Web->Kb) dataset is collected by the World Wide Knowledge Base
    (Web->Kb) project of the CMU text learning group. These pages were collected from computer
    science departments of various universities in 1997, manually classified into seven different
    classes: student, faculty, staff, department, course, project, and other. The classes Department
    and Staff is discarded, because there were only a few pages from each university. The class Other
    is discarded, because pages were very different among this class. The total number of training
    samples is 2,785 and testing 1,383.
    After preprocessing, the total number of training samples is 2,785 and testing samples 1,383.
    The min length of text about train data is 1, max length is 20,628, average length is 134; the min
    length of text about test data is 1, max length is 2,082, average length is 136.

    **Reference:** http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/

    Example:
        >>> train, test = webkb_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'student',
          'text': 'brian comput scienc depart univers wisconsin dayton street madison offic...'}
         {
          'label': 'student',
          'text': 'denni swanson web page mail pop uki offic hour comput lab offic anderson...'}]
        >>> test[0:2]
        [{
          'label': 'student',
          'text': 'eric homepag eric wei tsinghua physic fudan genet'},
         {
          'label': 'course',
          'text': 'comput system perform evalu model new sept assign due oct postscript text...'}]
    """

    share_id = '1oqcl2N0kDoBlHo_hFgKc_MaSvs0ny1t7' if preprocessing else '1rDpNXjbtKQzRepooh1QwEg2s_NgwnOOu'
    return imdb_dataset(directory, 'webkb', preprocessing, verbose=verbose, text_length=text_length, share_id=share_id)


def cade_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the Cade12 dataset (Version 1).

    The Cade12 dataset is corresponding to a subset of web pages extracted from the CADÊ Web Directory,
    which points to Brazilian web pages classified by human experts. The total number of training
    samples is 27,322 and testing 13,661.
    After preprocessing, the total number of training samples is 27,322 and testing samples 13,661.
    The min length of text about train data is 2, max length is 22,352, average length is 119; the min
    length of text about test data is 2, max length is 15,318, average length is 112.

    **Reference:** http://www.cade.com.br/

    Example:
        >>> train, test = cade_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': '08_cultura',
          'text': 'br br email arvores arvores http www apoio mascote natureza vida links foram...'}
         {
          'label': '02_sociedade',
          'text': 'page frames browser support virtual araraquara shop'}]
        >>> test[0:2]
        [{
          'label': '02_sociedade',
          'text': 'dezembro envie mail br manutencao funcionarios funcionarios funcionarios...'},
         {
          'label': '07_internet',
          'text': 'auto sao pagina br br computacao rede internet internet internet internet...'}]
    """

    share_id = '13CwKytxKlvMP6FW9iOCOMvmKlm5YWD-k' if preprocessing else '1GlxPC66_ECj4YRInkIYQXkmXCyOLFrsu'
    return imdb_dataset(directory, 'cade', preprocessing, verbose=verbose, text_length=text_length, share_id=share_id)


def dbpedia_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the DBPedia Ontology Classification dataset (Version 2).

    The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes
    from DBpedia 2014. They are listed in classes.txt. From each of these 14 ontology classes, we
    randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size
    of the training dataset is 560,000 and testing dataset 70,000.
    After preprocessing, the total number of training samples is 560,000 and testing samples 70,000.
    The min length of text about train data is 3, max length is 2,780, average length is 64; the min
    length of text about test data is 4, max length is 930, average length is 64.

    **Reference:** http://dbpedia.org

    Example:
        >>> train, test = dbpedia_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'Company',
          'text': 'e . d . abbott ltd abbott of farnham e d abbott limited was a british...'},
         {
          'label': 'Company',
          'text': 'schwan - stabilo schwan - stabilo is a german maker of pens for writing...'}]
        >>> test[0:2]
        [{
          'label': 'Company',
          'text': 'ty ku ty ku / ta ɪ ku ː / is an american alcoholic beverage company that...'},
         {
          'label': 'Company',
          'text': 'odd lot entertainment oddlot entertainment founded in 2 0 0 1 by longtime...'}]
    """

    share_id = '1egq6UCaaqeZOq7siitXEIfIFwYUjFjnP' if preprocessing else '1oB5-fQWMEz6RgIL9R9fT9P7ZkLNrob9s'
    return imdb_dataset(directory, 'dbpedia', preprocessing, verbose=verbose, text_length=text_length,
                        share_id=share_id)


def agnews_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the AG's News Topic Classification dataset (Version 3).

    The AG's news topic classification dataset is constructed by choosing 4 largest classes from
    the original corpus. Each class contains 30,000 training samples and 1,900 testing samples.
    The total number of training samples is 120,000 and testing 7,600.
    After preprocessing, the total number of training samples is 120,000 and testing samples 7,600.
    The min length of text about train data is 13, max length is 354, average length is 49; the min
    length of text about test data is 15, max length is 250, average length is 48.

    **Reference:** http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html

    Example:
        >>> train, test = agnews_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'Business',
          'text': 'wall st . bears claw back into the black ( reuters ) reuters - short - sellers...'},
         {
          'label': 'Business',
          'text': 'carlyle looks toward commercial aerospace ( reuters ) reuters - private investment...'}]
        >>> test[0:2]
        [{
          'label': 'Business',
          'text': 'fears for t n pension after talks unions representing workers at turner newall...'},
         {
          'label': 'Sci/Tech',
          'text': 'the race is on : second private team sets launch date for human spaceflight...'}]
    """

    share_id = '153R49C-JY8NDmRwc7bikZvU3EEEjKRs2' if preprocessing else '1pSX-jbwlGX5tDNAjzmxlBeCwNwzPXf42'
    return imdb_dataset(directory, 'agnews', preprocessing, verbose=verbose, text_length=text_length, share_id=share_id)


def yahoo_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the Yahoo! Answers Topic Classification dataset (Version 2).

    The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories.
    Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number
    of training samples is 1,400,000 and testing samples 60,000 in this dataset.
    After preprocessing, the total number of training samples is 1,400,000 and testing samples 60,000.
    The min length of text about train data is 2, max length is 4,044, average length is 118; the min
    length of text about test data is 3, max length is 4,017, average length is 119.

    **Reference:** https://webscope.sandbox.yahoo.com/catalog.php?datatype=l

    Example:
        >>> train, test = yahoo_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'Computers & Internet',
          'text': 'why doesn ' t an optical mouse work on a glass table ? or even on some surfaces...'},
         {
          'label': 'Sports',
          'text': 'what is the best off - road motorcycle trail ? long - distance trail throughout...'}]
        >>> test[0:2]
        [{
          'label': 'Family & Relationships',
          'text': 'what makes friendship click ? how does the spark keep going ? good communication...'},
         {
          'label': 'Science & Mathematics',
          'text': 'why does zebras have stripes ? what is the purpose or those stripes ? who do they...'}]
    """

    share_id = '1LS7iQM3qMofMCVlm08LfniyqXsdhFdnn' if preprocessing else '1REvRvHgeW5FQ3eHVgW3Hw7Mdoi2ZO4BK'
    return imdb_dataset(directory, 'yahoo', preprocessing, verbose=verbose, text_length=text_length, share_id=share_id)


def sogou_dataset(directory='data/', preprocessing=False, verbose=False, text_length=2700):
    """
    Load the Sogou News Topic Classification dataset (Version 3).

    The Sogou news topic classification dataset is constructed by manually labeling each news article
    according to its URL, which represents roughly the categorization of news in their websites. We
    chose 5 largest categories for the dataset, each having 90,000 samples for training and 12,000 for
    testing. The Pinyin texts are converted using pypinyin combined with jieba Chinese segmentation
    system. In total there are 450,000 training samples and 60,000 testing samples.
    After preprocessing, the total number of training samples is 450,000 and testing samples 60,000.
    The min length of text about train data is 2, max length is 42,695, average length is 612; the min
    length of text about test data is 3, max length is 64,651, average length is 616.

    **Reference:** http://www.sogou.com/labs/dl/ca.html and http://www.sogou.com/labs/dl/cs.html

    Example:
        >>> train, test = sogou_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': 'automobile',
          'text': '2 0 0 8 di4 qi1 jie4 qi1ng da3o guo2 ji4 che1 zha3n me3i nv3 mo2 te4 2 0 0 8...'}
         {
          'label': 'automobile',
          'text': 'zho1ng hua2 ju4n jie2 frv ya4o shi tu2 we2i zho1ng hua2 ju4n jie2 frv ya4o shi .'}]
        >>> test[0:2]
        [{
          'label': 'sports',
          'text': 'ti3 ca1o shi4 jie4 be1i : che2ng fe1i na2 pi2ng he2ng mu4 zi4 yo2u ca1o ji1n...'},
         {
          'label': 'automobile',
          'text': 'da3o ha2ng du2 jia1 ti2 go1ng me3i ri4 ba4o jia4 re4 xia4n : 0 1 0 - 6 4 4 3...'}]
    """

    share_id = '1HbJHzIacbQt7m-IRZzv8nRaSubSrYdip' if preprocessing else '10ue65ROxzrr0RN2QHNLNx_qerYEm4j-Z'
    return imdb_dataset(directory, 'sogou', preprocessing, verbose=verbose, text_length=text_length, share_id=share_id)


def yelp_dataset(directory='data/', preprocessing=False, fine_grained=False, verbose=False, text_length=2700):
    """
    Load the Yelp Review Full Star or Yelp Review Polarity dataset (Version 1).

    The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3
    and 4 positive. For each polarity 280,000 training samples and 19,000 testing samples are take
    randomly. In total there are 560,000 training samples and 38,000 testing samples. Negative
    polarity is class 1, and positive class 2.
    The Yelp reviews full star dataset is constructed by randomly taking 130,000 training samples
    and 10,000 testing samples for each review star from 1 to 5. In total there are 650,000 training
    samples and 50,000 testing samples.
    After preprocessing, the total number of training samples is 560,000 and testing samples 38,000.
    The min length of text about train data is 1, max length is 1,491, average length is 162; the min
    length of text about test data is 1, max length is 1,311, average length is 162. (polarity)
    After preprocessing, the total number of training samples is 650,000 and testing samples 50,000.
    The min length of text about train data is 1, max length is 1,332, average length is 164; the min
    length of text about test data is 1, max length is 1,491, average length is 164. (full)

    **Reference:** http://www.yelp.com/dataset_challenge

    Example:
        >>> train, test = yelp_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': '1',
          'text': 'unfortunately , the frustration of being dr . goldberg ' s patient is a repeat...'}
         {
          'label': '2',
          'text': 'been going to dr . goldberg for over 1 0 years . i think i was one of his 1...'}]
        >>> test[0:2]
        [{
          'label': '2',
          'text': 'contrary to other reviews , i have zero complaints about the service or the prices...'},
         {
          'label': '1',
          'text': 'last summer i had an appointment to get new tires and had to wait a super long time...'}]
    """

    share_id = '1ecOuyAhT-MjXQiueRHqS9LnY0CV0HQYd' if preprocessing else '1GYu-TT-H_5jXJpWe_tvOHXTXCyxV55-J'
    return imdb_dataset(directory, 'yelp', preprocessing, fine_grained, verbose, text_length, share_id)


def amazon_dataset(directory='data/', preprocessing=False, fine_grained=False, verbose=False, text_length=2700):
    """
    Load the Amazon Review Full Score or Amazon Review Polaridy dataset (Version 3).

    The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative,
    and 4 and 5 as positive. For each polarity 1,800,000 training samples and 200,000 testing samples
    are take randomly. In total there are 3,600,000 training samples and 400,000 testing samples.
    Negative polarity is class 1, and positive class 2.
    The Amazon reviews full score dataset is constructed by randomly taking 600,000 training samples
    and 130,000 testing samples for each review score from 1 to 5. In total there are 3,000,000
    training samples and 650,000 testing samples.
    After preprocessing, the total number of training samples is 3,600,000 and testing samples 400,000.
    The min length of text about train data is 2, max length is 986, average length is 95; the min
    length of text about test data is 14, max length is 914, average length is 95. (polarity)
    After preprocessing, the total number of training samples is 3,000,000 and testing samples 650,000.
    The min length of text about train data is 2, max length is 781, average length is 97; the min
    length of text about test data is 12, max length is 931, average length is 97. (full)

    **Reference:** http://jmcauley.ucsd.edu/data/amazon/

    Example:
        >>> train, test = amazon_dataset(preprocessing=True)
        >>> train[0:2]
        [{
          'label': '2',
          'text': 'stuning even for the non - gamer this sound track was beautiful ! it paints...'}
         {
          'label': '2',
          'text': 'the best soundtrack ever to anything . i ' m reading a lot of reviews saying...'}]
        >>> test[0:2]
        [{
          'label': '2',
          'text': 'great cd my lovely pat has one of the great voices of her generation . i have...'},
         {
          'label': '2',
          'text': 'one of the best game music soundtracks - for a game i didn ' t really play...'}]
    """

    share_id = '1BSqCU6DwIVD1jllbsz9ueudu3tSfomzY' if preprocessing else '1pYKd_h6OIzwVOVwocqEEOuuxYOe16HlU'
    return imdb_dataset(directory, 'amazon', preprocessing, fine_grained, verbose, text_length, share_id)
