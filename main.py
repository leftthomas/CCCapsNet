import argparse

import torch
import torch.nn as nn
import torchnet as tnt
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

from model import Model
from utils import get_iterator


def processor(sample):
    data, labels, training = sample

    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    data = Variable(data)
    labels = Variable(labels)

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].data, state['sample'][1])
    confusion_meter.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])

    # learning rate scheduler
    scheduler.step(meter_loss.value()[0], epoch=state['epoch'])

    reset_meters()

    engine.test(processor, get_iterator('TREC', False, BATCH_SIZE))

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(confusion_meter.value())

    print('[Epoch %d] Testing Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    torch.save(model.state_dict(), 'epochs/%d.pth' % (state['epoch']))

    # visualization
    # test_image, _ = next(iter(get_iterator('TREC', False, 25)))
    # test_image_logger.log(make_grid(test_image, nrow=5, normalize=True).numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Classfication')
    parser.add_argument('--data_type', default='TREC', type=str, choices=['TREC'], help='dataset type')
    parser.add_argument('--batch_size', default=100, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()
    DATA_TYPE = opt.data_type
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    model = Model()
    loss_criterion = nn.BCEWithLogitsLoss()
    if torch.cuda.is_available():
        model.cuda()
        loss_criterion.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(6, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion Matrix'})
    # test_image_logger = VisdomLogger('image', opts={'title': 'Test Image', 'width': 371, 'height': 335})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator('TREC', True, BATCH_SIZE), maxepoch=NUM_EPOCHS, optimizer=optimizer)
