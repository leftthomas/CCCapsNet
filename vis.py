import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding')
    parser.add_argument('--data_type', default='imdb', type=str,
                        choices=['imdb', 'newsgroups', 'reuters', 'webkb', 'cade', 'dbpedia', 'agnews', 'yahoo',
                                 'sogou', 'yelp', 'amazon'], help='dataset type')
    parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                    'reuters, yelp and amazon')
    parser.add_argument('--text_length', default=5000, type=int, help='the number of words about the text to load')
    parser.add_argument('--routing_type', default='k_means', type=str, choices=['k_means', 'dynamic'],
                        help='routing type, it only works for capsule classifier')
    parser.add_argument('--loss_type', default='mf', type=str,
                        choices=['margin', 'focal', 'cross', 'mf', 'mc', 'fc', 'mfc'], help='loss type')
    parser.add_argument('--embedding_type', default='cwc', type=str, choices=['cwc', 'cc', 'normal'],
                        help='embedding type')
    parser.add_argument('--classifier_type', default='capsule', type=str, choices=['capsule', 'linear'],
                        help='classifier type')
    parser.add_argument('--embedding_size', default=64, type=int, help='embedding size')
    parser.add_argument('--num_codebook', default=8, type=int,
                        help='codebook number, it only works for cwc and cc embedding')
    parser.add_argument('--num_codeword', default=None, type=int,
                        help='codeword number, it only works for cwc and cc embedding')
    parser.add_argument('--hidden_size', default=128, type=int, help='hidden size')
    parser.add_argument('--in_length', default=8, type=int,
                        help='in capsule length, it only works for capsule classifier')
    parser.add_argument('--out_length', default=16, type=int,
                        help='out capsule length, it only works for capsule classifier')
    parser.add_argument('--num_iterations', default=3, type=int,
                        help='routing iterations number, it only works for capsule classifier')
    parser.add_argument('--drop_out', default=0.5, type=float, help='drop_out rate of GRU layer')
    parser.add_argument('--batch_size', default=30, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=10, type=int, help='train epochs number')
    parser.add_argument('--num_steps', default=100, type=int, help='test steps number')
    parser.add_argument('--load_model_weight', default=None, type=str, help='saved model weight to load')
