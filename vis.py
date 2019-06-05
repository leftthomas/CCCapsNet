import argparse

import torch
import torch.backends.cudnn as cudnn

from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding')
    parser.add_argument('--load_model_weight', default=None, type=str, help='saved model weight to load')

    opt = parser.parse_args()
    DATA_TYPE, FINE_GRAINED, TEXT_LENGTH = opt.data_type, opt.fine_grained, opt.text_length
    MODEL_WEIGHT = opt.load_model_weight
    # prepare dataset
    sentence_encoder, label_encoder, train_dataset, test_dataset = load_data(DATA_TYPE, preprocessing=True,
                                                                             fine_grained=FINE_GRAINED, verbose=True,
                                                                             text_length=TEXT_LENGTH)
    model = torch.load('epochs/' + MODEL_WEIGHT)
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to('cuda'), True

    vocabs, codes = [], []
    if model.embedding_type == 'normal':
        embedding = model.embedding.weight
        for index, vocab in enumerate(sentence_encoder.vocab):
            vocabs.append({vocab: embedding[index]})
    else:
        embedding = model.embedding
        embedding.return_code = True
        for vocab in sentence_encoder.vocab:
            out, code = embedding(torch.Tensor([[vocab]]))
            vocabs.append({vocab: out.squeeze()})
            codes.append({vocab: code.squeeze()})
