import os
import sys

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def cade_dataset(directory='data/', train=False, test=False, extracted_name='cade',
                 check_files=['cade/cade-train-stemmed.txt'],
                 url='https://drive.google.com/uc?export=download&id=1psVDSlbSQuEnEtPE8L8U7UH5mhwmAv_m'):
    download_file_maybe_extract(url=url, directory=directory, filename='cade.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in
              [(train, 'cade-train-stemmed.txt'), (test, 'cade-test-stemmed.txt')] if requested]
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

    train_file = 'data/cade_train.txt'
    train_f = open(train_file, 'w')
    for train_data in ret[0]:
        train_f.write(train_data['label'] + '\t' + train_data['text'] + '\n')
    train_f.close()
    test_file = 'data/cade_test.txt'
    test_f = open(test_file, 'w')
    for test_data in ret[1]:
        test_f.write(test_data['label'] + '\t' + test_data['text'] + '\n')
    test_f.close()

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
