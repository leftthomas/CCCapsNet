import os
import sys

import numpy as np
import pandas as pd
from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def sogou_dataset(directory='data/', train=False, test=False, extracted_name='sogou_news',
                  check_files=['sogou_news/readme.txt'],
                  url='https://drive.google.com/uc?export=download&id=13WPe20BmrI_a9vgRa91PxfTrbBcR-1Ee'):
    download_file_maybe_extract(url=url, directory=directory, filename='sogou_news.tar.gz', check_files=check_files)

    ret = []
    splits = [file_name for (requested, file_name) in [(train, 'train.csv'), (test, 'test.csv')] if requested]
    index_to_label = []
    with open(os.path.join(directory, extracted_name, 'classes.txt'), 'r', encoding='utf-8') as foo:
        for line in foo.readlines():
            line = line.rstrip('\n')
            index_to_label.append(line)
    for file_name in splits:
        csv_file = np.array(pd.read_csv(os.path.join(directory, extracted_name, file_name), header=None)).tolist()
        examples = []
        text_min_length = sys.maxsize
        text_max_length = 0
        for data in csv_file:
            label, title, description = index_to_label[int(data[0]) - 1], data[1], data[2]

            if (not isinstance(title, str)) and (not isinstance(description, str)):
                continue
            else:
                if isinstance(title, str) and (not isinstance(description, str)):
                    text = title
                elif isinstance(description, str) and (not isinstance(title, str)):
                    text = description
                else:
                    text = title + ' ' + description
                # text = text_preprocess(text)
                if len(text.split()) == 0:
                    continue
                else:
                    if len(text.split()) > text_max_length:
                        text_max_length = len(text.split())
                    if len(text.split()) < text_min_length:
                        text_min_length = len(text.split())
            examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))
        print('text_min_length:' + str(text_min_length))
        print('text_max_length:' + str(text_max_length))

    train_file = 'data/sogou_train.txt'
    train_f = open(train_file, 'w')
    for train_data in ret[0]:
        train_f.write(train_data['label'] + '\t' + train_data['text'] + '\n')
    train_f.close()
    test_file = 'data/sogou_test.txt'
    test_f = open(test_file, 'w')
    for test_data in ret[1]:
        test_f.write(test_data['label'] + '\t' + test_data['text'] + '\n')
    test_f.close()

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
