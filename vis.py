import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn

from model import Model
from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding and Code')
    parser.add_argument('--model_weight', default=None, type=str, help='saved model weight to load')
    parser.add_argument('--routing_type', default='k_means', type=str, choices=['k_means', 'dynamic'],
                        help='routing type, it only works for capsule classifier')
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
    parser.add_argument('--num_repeat', default=10, type=int,
                        help='gumbel softmax repeat number, it only works for cc embedding')
    parser.add_argument('--drop_out', default=0.5, type=float, help='drop_out rate of GRU layer')

    opt = parser.parse_args()
    MODEL_WEIGHT, ROUTING_TYPE, EMBEDDING_SIZE = opt.model_weight, opt.routing_type, opt.embedding_size
    NUM_CODEBOOK, NUM_CODEWORD, HIDDEN_SIZE = opt.num_codebook, opt.num_codeword, opt.hidden_size
    IN_LENGTH, OUT_LENGTH, NUM_ITERATIONS, DROP_OUT = opt.in_length, opt.out_length, opt.num_iterations, opt.drop_out
    NUM_REPEAT = opt.num_repeat
    configs = MODEL_WEIGHT.split('_')
    if len(configs) == 4:
        DATA_TYPE, EMBEDDING_TYPE, CLASSIFIER_TYPE, TEXT_LENGTH = configs
        FINE_GRAINED, TEXT_LENGTH = False, int(TEXT_LENGTH.split('.')[0])
    else:
        DATA_TYPE, _, EMBEDDING_TYPE, CLASSIFIER_TYPE, TEXT_LENGTH = configs
        FINE_GRAINED, TEXT_LENGTH = True, int(TEXT_LENGTH.split('.')[0])

    data_name = '{}_fine-grained'.format(DATA_TYPE) if FINE_GRAINED else DATA_TYPE

    print('Loading {} dataset'.format(data_name))
    # get sentence encoder
    sentence_encoder, label_encoder, _, _ = load_data(DATA_TYPE, preprocessing=True, fine_grained=FINE_GRAINED,
                                                      verbose=True, text_length=TEXT_LENGTH)
    VOCAB_SIZE, NUM_CLASS = sentence_encoder.vocab_size, label_encoder.vocab_size

    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, NUM_CODEBOOK, NUM_CODEWORD, HIDDEN_SIZE, IN_LENGTH, OUT_LENGTH,
                  NUM_CLASS, ROUTING_TYPE, EMBEDDING_TYPE, CLASSIFIER_TYPE, NUM_ITERATIONS, NUM_REPEAT, DROP_OUT)
    model.load_state_dict(torch.load('epochs/{}'.format(MODEL_WEIGHT), map_location='cpu'))
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to('cuda'), True

    model.eval()
    print('Generating embedding and code for {} dataset'.format(data_name))
    with torch.no_grad():
        if EMBEDDING_TYPE == 'normal':
            vocabs = model.embedding.weight.detach().cpu().numpy()
            codes = torch.ones(1, 1, sentence_encoder.vocab_size)
        else:
            embedding = model.embedding
            embedding.return_code = True
            data = torch.arange(sentence_encoder.vocab_size).view(1, -1)
            if torch.cuda.is_available():
                data = data.to('cuda')
            out, code = embedding(data)
            # [num_embeddings, embedding_dim], ([num_embeddings, num_codebook, num_codeword], [1, 1, num_embeddings])
            vocabs, codes = out.squeeze(dim=0).detach().cpu().numpy(), code.squeeze(dim=0).detach().cpu()

    print('Plotting code usage for {} dataset'.format(data_name))
    reduced_codes = codes.sum(dim=0).float()
    c_max, c_min = reduced_codes.max().item(), reduced_codes.min().item()
    f, ax = plt.subplots(figsize=(10, 5))
    heat_map = sns.heatmap(reduced_codes.numpy(), vmin=c_min, vmax=c_max, annot=True, fmt='.2f', ax=ax)
    ax.set_title('Code usage of {} embedding for {} dataset'.format(EMBEDDING_TYPE, data_name))
    ax.set_xlabel('codeword')
    ax.set_ylabel('codebook')
    f.savefig('results/{}_{}_code.jpg'.format(data_name, EMBEDDING_TYPE))
