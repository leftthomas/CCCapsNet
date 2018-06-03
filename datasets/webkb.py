import os

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def webkb_dataset(directory='data/', train=False, test=False, extracted_name='webkb',
                  check_files=['webkb/webkb-train-stemmed.txt'],
                  url='https://drive.google.com/uc?export=download&id=1psVDSlbSQuEnEtPE8L8U7UH5mhwmAv_m'):
    """
    Load the World Wide Knowledge Base (Web->Kb) dataset (Version 1).

    The World Wide Knowledge Base (Web->Kb) dataset is collected by the World Wide
    Knowledge Base (Web->Kb) project of the CMU text learning group. These pages
    were collected from computer science departments of various universities in 1997,
    manually classified into seven different classes: student, faculty, staff,
    department, course, project, and other. The classes Department and Staff is
    discarded, because there were only a few pages from each university. The class
    Other is discarded, because pages were very different among this class. The total
    number of training samples is 2,803 and testing 1,396.
    The min length of text about train data is 15, max length of it is 594; The min length
    of text about test data is 42, max length of it is 497.

    **Reference:** http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/

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
        >>> from datasets import webkb_dataset
        >>> train = webkb_dataset(train=True)
        >>> train[0:2]
        [{
          'label': 'student',
          'text': 'brian comput scienc depart univers wisconsin dayton street madison offic...'}
         {
          'label': 'student',
          'text': 'denni swanson web page mail pop uki offic hour comput lab offic anderson...'}]
    """
    download_file_maybe_extract(url=url, directory=directory, filename='webkb.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in
              [(train, 'webkb-train-stemmed.txt'), (test, 'webkb-test-stemmed.txt')] if requested]
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
