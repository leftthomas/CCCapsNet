import argparse

import torch
import torch.backends.cudnn as cudnn

from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding')
    parser.add_argument('--data_type', default='imdb', type=str,
                        choices=['imdb', 'newsgroups', 'reuters', 'webkb', 'cade', 'dbpedia', 'agnews', 'yahoo',
                                 'sogou', 'yelp', 'amazon'], help='dataset type')
    parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                    'reuters, yelp and amazon')
    parser.add_argument('--text_length', default=5000, type=int, help='the number of words about the text to load')
    parser.add_argument('--load_model_weight', default=None, type=str, help='saved model weight to load')

    opt = parser.parse_args()
    DATA_TYPE, FINE_GRAINED, TEXT_LENGTH = opt.data_type, opt.fine_grained, opt.text_length
    MODEL_WEIGHT = opt.load_model_weight
    # prepare dataset
    VOCAB_SIZE, NUM_CLASS, train_dataset, test_dataset = load_data(DATA_TYPE, preprocessing=True,
                                                                   fine_grained=FINE_GRAINED, verbose=True,
                                                                   text_length=TEXT_LENGTH)
    print("[!] vocab_size: {}, num_class: {}".format(VOCAB_SIZE, NUM_CLASS))
    model = torch.load('epochs/' + MODEL_WEIGHT)
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to('cuda'), True
