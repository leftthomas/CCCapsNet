import argparse

import torch
import torchnet as tnt
from capsule_layer.optim import MultiStepRI
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchnlp.samplers import BucketBatchSampler
from tqdm import tqdm

from model import Model
from utils import load_data, MarginLoss, collate_fn


def processor(sample):
    data, label, training = sample
    label = torch.eye(num_class).index_select(dim=0, index=label)
    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()
    data = Variable(data)
    label = Variable(label)

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, label)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    meter_confusion.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].data, state['sample'][1])
    meter_confusion.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])

    reset_meters()

    test_sampler = BucketBatchSampler(test_dataset, BATCH_SIZE, False, sort_key=lambda row: len(row['text']))
    test_iterator = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    engine.test(processor, test_iterator)

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(meter_confusion.value())

    # save best model
    # pay attention, it's a global value
    global best_acc
    if meter_accuracy.value()[0] > best_acc:
        best_acc = meter_accuracy.value()[0]
        if FINE_GRAINED:
            torch.save(model.state_dict(), 'epochs/%s.pth' % (DATA_TYPE + '_fine_grained'))
        else:
            torch.save(model.state_dict(), 'epochs/%s.pth' % DATA_TYPE)

    print('[Epoch %d] Testing Loss: %.4f Accuracy: %.2f%% Best Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], best_acc))

    # scheduler routing iterations
    scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Text Classification')
    parser.add_argument('--data_type', default='imdb', type=str,
                        choices=['imdb', 'newsgroups', 'reuters', 'webkb', 'dbpedia', 'agnews', 'yahoo', 'yelp',
                                 'amazon'], help='dataset type')
    parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                    'reuters, yelp and amazon')
    parser.add_argument('--num_iterations', default=1, type=int, help='initial routing iterations number')
    parser.add_argument('--batch_size', default=30, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()
    DATA_TYPE = opt.data_type
    FINE_GRAINED = opt.fine_grained
    NUM_ITERATIONS = opt.num_iterations
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # prepare dataset
    vocab_size, num_class, train_dataset, test_dataset = load_data(DATA_TYPE, FINE_GRAINED)
    print("[!] vocab_size: {}, num_class: {}".format(vocab_size, num_class))
    train_sampler = BucketBatchSampler(train_dataset, BATCH_SIZE, False, sort_key=lambda row: len(row['text']))
    train_iterator = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

    model = Model(vocab_size, num_class=num_class, num_iterations=NUM_ITERATIONS)
    loss_criterion = MarginLoss()
    if torch.cuda.is_available():
        model.cuda()
        loss_criterion.cuda()

    scheduler = MultiStepRI(model, milestones=[10, 30, 70], verbose=True)
    optimizer = Adam(model.parameters())
    print("# trainable parameters:", sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_confusion = tnt.meter.ConfusionMeter(num_class, normalized=True)

    # record current best test accuracy
    best_acc = 0

    if FINE_GRAINED:
        env_name = DATA_TYPE + '_fine_grained'
    else:
        env_name = DATA_TYPE
    train_loss_logger = VisdomPlotLogger('line', env=env_name, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=env_name, opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=env_name, opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', env=env_name, opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', env=env_name, opts={'title': 'Confusion Matrix'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_iterator, maxepoch=NUM_EPOCHS, optimizer=optimizer)
