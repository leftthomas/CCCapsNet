from torchtext.datasets import TREC


class BatchWrapper:
    def __init__(self, dl, x_var, y_var):
        self.dl, self.x_var, self.y_var = dl, x_var, y_var

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)
            y = getattr(batch, self.y_var)
            yield [x, y]

    def __len__(self):
        return len(self.dl)


def get_iterator(data_type, mode, batch_size=100):
    if data_type == 'TREC':
        train_iter, test_iter = TREC.iters(batch_size=batch_size, device=-1, root='data')
        if mode:
            return BatchWrapper(train_iter, x_var='text', y_var='label')
        else:
            return BatchWrapper(test_iter, x_var='text', y_var='label')
    else:
        raise NotImplementedError()
