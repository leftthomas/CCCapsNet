import torchtext.data as data
import torchtext.datasets as datasets


def _iters(data_type, batch_size, fine_grained):
    text = data.Field(sequential=True)
    label = data.LabelField()

    train, val, test = datasets.SST.splits(text, label, root='data', fine_grained=fine_grained)

    text.build_vocab(train, vectors='glove.6B.300d')
    label.build_vocab(train)

    return text, label, data.BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False)


def load_data(data_type, batch_size, fine_grained=True):
    text, label, (train_iter, val_iter, test_iter) = _iters(data_type, batch_size, fine_grained)

    data_info = {'vocab_size': len(text.vocab), 'num_class': 5 if fine_grained else 3, 'text': text}

    return train_iter, val_iter, test_iter, data_info
