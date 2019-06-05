import argparse

import torch
import torch.backends.cudnn as cudnn

from utils import load_data

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
    # get sentence encoder
    sentence_encoder, _, _, _ = load_data(DATA_TYPE, preprocessing=True, fine_grained=FINE_GRAINED, verbose=True,
                                          text_length=TEXT_LENGTH)
    model = torch.load('epochs/' + MODEL_WEIGHT)
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to('cuda'), True

    model.eval()
    with torch.no_grad():
        vocabs, codes = {}, {}
        if EMBEDDING_TYPE == 'normal':
            embedding = model.embedding.weight
            for index, vocab in enumerate(sentence_encoder.vocab):
                vocabs[vocab] = embedding[index]
        else:
            embedding = model.embedding
            embedding.return_code = True
            for index, vocab in enumerate(sentence_encoder.vocab):
                data = torch.tensor([[index]])
                if torch.cuda.is_available():
                    data = data.to('cuda')
                out, code = embedding(data)
                vocabs[vocab], codes[vocab] = out.squeeze(dim=0).squeeze(dim=0), code.squeeze(dim=0)
