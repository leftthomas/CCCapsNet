import os
import sys

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
    number of training samples is 2,785 and testing 1,383.

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
    """
    download_file_maybe_extract(url=url, directory=directory, filename='webkb.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in
              [(train, 'webkb-train-stemmed.txt'), (test, 'webkb-test-stemmed.txt')] if requested]
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

    train_file = 'data/webkb_train.txt'
    train_f = open(train_file, 'w')
    for train_data in ret[0]:
        train_f.write(train_data['label'] + '\t' + train_data['text'] + '\n')
    train_f.close()
    test_file = 'data/webkb_test.txt'
    test_f = open(test_file, 'w')
    for test_data in ret[1]:
        test_f.write(test_data['label'] + '\t' + test_data['text'] + '\n')
    test_f.close()

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
