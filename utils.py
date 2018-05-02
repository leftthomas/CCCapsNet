from torchtext.datasets import TREC


def get_iterator(data_type, mode, batch_size=100):
    if data_type == 'TREC':
        train_data, test_data = TREC.iters(batch_size=batch_size, device=-1, root='data')[1]
        return train_data if mode else test_data
    else:
        raise NotImplementedError()
