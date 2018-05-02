from torchtext.datasets import TREC


def get_iterator(data_type, batch_size=64):
    if data_type == 'TREC':
        return TREC.iters(batch_size=batch_size, device=-1, root='data')
    else:
        raise NotImplementedError()
