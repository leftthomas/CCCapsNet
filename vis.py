import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn

from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding and Code')
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
