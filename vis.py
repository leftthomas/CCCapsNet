import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE

from utils import load_data


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], label[i][0], color=plt.cm.Set1(i / len(label)),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding')
    parser.add_argument('--load_model_weight', default=None, type=str, help='saved model weight to load')

    opt = parser.parse_args()
    MODEL_WEIGHT = opt.load_model_weight
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
    sentence_encoder, _, _, _ = load_data(DATA_TYPE, preprocessing=True, fine_grained=FINE_GRAINED, verbose=True,
                                          text_length=TEXT_LENGTH)
    model = torch.load('epochs/' + MODEL_WEIGHT)
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to('cuda'), True

    model.eval()
    print('Generating t-SNE embedding for {} dataset'.format(data_name))
    with torch.no_grad():
        if EMBEDDING_TYPE == 'normal':
            vocabs = model.embedding.weight.detach().cpu().numpy()
            codes = torch.ones(sentence_encoder.vocab_size, 1, 1).numpy()
        else:
            embedding = model.embedding
            embedding.return_code = True
            data = torch.arange(sentence_encoder.vocab_size).view(1, -1)
            if torch.cuda.is_available():
                data = data.to('cuda')
            out, code = embedding(data)
            # [num_embeddings, embedding_dim], ([num_embeddings, num_codebook, num_codeword], [num_embeddings, 1, 1])
            vocabs, codes = out.squeeze(dim=0).detach().cpu().numpy(), code.squeeze(dim=0).detach().cpu().numpy()

        indexes = []
        for word in ['dog', 'dogs', 'cat', 'cats', 'penguin', 'penguins', 'man', 'woman', 'men', 'women', 'king',
                     'queen', 'go', 'went', 'gone', 'homes', 'cruises', 'motel', 'basketball', 'softball', 'enough',
                     'hardly', 'unfortunately', 'fortunately', 'obviously', 'toronto', 'oakland']:
            if word in sentence_encoder.vocab:
                indexes.append(sentence_encoder.vocab.index(word))
        if len(indexes) == 0:
            raise IndexError('Make sure the vocabs contain these words')
        reduced_vocabs, reduced_codes = vocabs[indexes], codes[indexes]
        result = TSNE(n_components=2, init='pca', random_state=0).fit_transform(reduced_vocabs)
        fig = plot_embedding(result, sentence_encoder.vocab.index(indexes), 't-SNE embedding of {}'.format(data_name))
        print('Plotting t-SNE embedding for {} dataset'.format(data_name))
        plt.savefig('results/{}_{}_tsne.jpg'.format(data_name, EMBEDDING_TYPE))
        # plt.show(fig)
