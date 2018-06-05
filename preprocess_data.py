import argparse

from original_datasets import imdb_dataset, ag_dataset, amazon_dataset, dbpedia_dataset, newsgroups_dataset, \
    reuters_dataset, \
    webkb_dataset, yahoo_dataset, yelp_dataset


def load_data(data_type, fine_grained):
    if data_type == 'IMDB':
        dataset = imdb_dataset(train=True, test=True)
    elif data_type == 'Newsgroups':
        dataset = newsgroups_dataset(train=True, test=True)
    elif data_type == 'Reuters':
        dataset = reuters_dataset(train=True, test=True, fine_grained=fine_grained)
    elif data_type == 'WebKB':
        dataset = webkb_dataset(train=True, test=True)
    elif data_type == 'DBPedia':
        dataset = dbpedia_dataset(train=True, test=True)
    elif data_type == 'AGNews':
        dataset = ag_dataset(train=True, test=True)
    elif data_type == 'YahooAnswers':
        dataset = yahoo_dataset(train=True, test=True)
    elif data_type == 'YelpReview':
        dataset = yelp_dataset(train=True, test=True, fine_grained=fine_grained)
    elif data_type == 'AmazonReview':
        dataset = amazon_dataset(train=True, test=True, fine_grained=fine_grained)
    else:
        raise ValueError('{} data type not supported.'.format(data_type))

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Text Datasets')
    parser.add_argument('--data_type', default='AGNews', type=str,
                        choices=['IMDB', 'Newsgroups', 'Reuters', 'WebKB', 'DBPedia', 'AGNews', 'YahooAnswers',
                                 'YelpReview', 'AmazonReview'], help='dataset type')
    parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                    'Reuters, YelpReview and AmazonReview')

    opt = parser.parse_args()
    DATA_TYPE = opt.data_type
    FINE_GRAINED = opt.fine_grained

    # prepare dataset
    datasets = load_data(DATA_TYPE, FINE_GRAINED)
