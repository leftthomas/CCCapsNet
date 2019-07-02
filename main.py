import argparse

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchnet as tnt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchnlp.samplers import BucketBatchSampler

from model import Model
from utils import load_data, MarginLoss, collate_fn, FocalLoss


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    meter_confusion.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Text Classification')
    parser.add_argument('--data_type', default='imdb', type=str,
                        choices=['imdb', 'newsgroups', 'reuters', 'webkb', 'cade', 'dbpedia', 'agnews', 'yahoo',
                                 'sogou', 'yelp', 'amazon'], help='dataset type')
    parser.add_argument('--fine_grained', action='store_true', help='use fine grained class or not, it only works for '
                                                                    'reuters, yelp and amazon')
    parser.add_argument('--text_length', default=5000, type=int, help='the number of words about the text to load')
    parser.add_argument('--routing_type', default='k_means', type=str, choices=['k_means', 'dynamic'],
                        help='routing type, it only works for capsule classifier')
    parser.add_argument('--loss_type', default='mf', type=str,
                        choices=['margin', 'focal', 'cross', 'mf', 'mc', 'fc', 'mfc'], help='loss type')
    parser.add_argument('--embedding_type', default='cwc', type=str, choices=['cwc', 'cc', 'normal'],
                        help='embedding type')
    parser.add_argument('--classifier_type', default='capsule', type=str, choices=['capsule', 'linear'],
                        help='classifier type')
    parser.add_argument('--embedding_size', default=64, type=int, help='embedding size')
    parser.add_argument('--num_codebook', default=8, type=int,
                        help='codebook number, it only works for cwc and cc embedding')
    parser.add_argument('--num_codeword', default=None, type=int,
                        help='codeword number, it only works for cwc and cc embedding')
    parser.add_argument('--hidden_size', default=128, type=int, help='hidden size')
    parser.add_argument('--in_length', default=8, type=int,
                        help='in capsule length, it only works for capsule classifier')
    parser.add_argument('--out_length', default=16, type=int,
                        help='out capsule length, it only works for capsule classifier')
    parser.add_argument('--num_iterations', default=3, type=int,
                        help='routing iterations number, it only works for capsule classifier')
    parser.add_argument('--num_repeat', default=10, type=int,
                        help='gumbel softmax repeat number, it only works for cc embedding')
    parser.add_argument('--drop_out', default=0.5, type=float, help='drop_out rate of GRU layer')
    parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=10, type=int, help='train epochs number')
    parser.add_argument('--num_steps', default=100, type=int, help='test steps number')
    parser.add_argument('--pre_model', default=None, type=str,
                        help='pre-trained model weight, it only works for routing_type experiment')

    opt = parser.parse_args()
    DATA_TYPE, FINE_GRAINED, TEXT_LENGTH = opt.data_type, opt.fine_grained, opt.text_length
    ROUTING_TYPE, LOSS_TYPE, EMBEDDING_TYPE = opt.routing_type, opt.loss_type, opt.embedding_type
    CLASSIFIER_TYPE, EMBEDDING_SIZE, NUM_CODEBOOK = opt.classifier_type, opt.embedding_size, opt.num_codebook
    NUM_CODEWORD, HIDDEN_SIZE, IN_LENGTH = opt.num_codeword, opt.hidden_size, opt.in_length
    OUT_LENGTH, NUM_ITERATIONS, DROP_OUT, BATCH_SIZE = opt.out_length, opt.num_iterations, opt.drop_out, opt.batch_size
    NUM_REPEAT, NUM_EPOCHS, NUM_STEPS, PRE_MODEL = opt.num_repeat, opt.num_epochs, opt.num_steps, opt.pre_model

    # prepare dataset
    sentence_encoder, label_encoder, train_dataset, test_dataset = load_data(DATA_TYPE, preprocessing=True,
                                                                             fine_grained=FINE_GRAINED, verbose=True,
                                                                             text_length=TEXT_LENGTH)
    VOCAB_SIZE, NUM_CLASS = sentence_encoder.vocab_size, label_encoder.vocab_size
    print("[!] vocab_size: {}, num_class: {}".format(VOCAB_SIZE, NUM_CLASS))
    train_sampler = BucketBatchSampler(train_dataset, BATCH_SIZE, False, sort_key=lambda row: len(row['text']))
    train_iterator = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    test_sampler = BucketBatchSampler(test_dataset, BATCH_SIZE * 2, False, sort_key=lambda row: len(row['text']))
    test_iterator = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    model = Model(VOCAB_SIZE, EMBEDDING_SIZE, NUM_CODEBOOK, NUM_CODEWORD, HIDDEN_SIZE, IN_LENGTH, OUT_LENGTH,
                  NUM_CLASS, ROUTING_TYPE, EMBEDDING_TYPE, CLASSIFIER_TYPE, NUM_ITERATIONS, NUM_REPEAT, DROP_OUT)
    if PRE_MODEL is not None:
        model_weight = torch.load('epochs/{}'.format(PRE_MODEL), map_location='cpu')
        model_weight.pop('classifier.weight')
        model.load_state_dict(model_weight, strict=False)

    if LOSS_TYPE == 'margin':
        loss_criterion = [MarginLoss(NUM_CLASS)]
    elif LOSS_TYPE == 'focal':
        loss_criterion = [FocalLoss()]
    elif LOSS_TYPE == 'cross':
        loss_criterion = [CrossEntropyLoss()]
    elif LOSS_TYPE == 'mf':
        loss_criterion = [MarginLoss(NUM_CLASS), FocalLoss()]
    elif LOSS_TYPE == 'mc':
        loss_criterion = [MarginLoss(NUM_CLASS), CrossEntropyLoss()]
    elif LOSS_TYPE == 'fc':
        loss_criterion = [FocalLoss(), CrossEntropyLoss()]
    else:
        loss_criterion = [MarginLoss(NUM_CLASS), FocalLoss(), CrossEntropyLoss()]
    if torch.cuda.is_available():
        model, cudnn.benchmark = model.to('cuda'), True

    if PRE_MODEL is None:
        optim_configs = [{'params': model.embedding.parameters(), 'lr': 1e-4 * 10},
                         {'params': model.features.parameters(), 'lr': 1e-4 * 10},
                         {'params': model.classifier.parameters(), 'lr': 1e-4}]
    else:
        for param in model.embedding.parameters():
            param.requires_grad = False
        for param in model.features.parameters():
            param.requires_grad = False
        optim_configs = [{'params': model.classifier.parameters(), 'lr': 1e-4}]
    optimizer = Adam(optim_configs, lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)

    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    # record statistics
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    # record current best test accuracy
    best_acc = 0
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_confusion = tnt.meter.ConfusionMeter(NUM_CLASS, normalized=True)

    # config the visdom figures
    if FINE_GRAINED and DATA_TYPE in ['reuters', 'yelp', 'amazon']:
        env_name = DATA_TYPE + '_fine_grained'
    else:
        env_name = DATA_TYPE
    loss_logger = VisdomPlotLogger('line', env=env_name, opts={'title': 'Loss'})
    accuracy_logger = VisdomPlotLogger('line', env=env_name, opts={'title': 'Accuracy'})
    train_confusion_logger = VisdomLogger('heatmap', env=env_name, opts={'title': 'Train Confusion Matrix'})
    test_confusion_logger = VisdomLogger('heatmap', env=env_name, opts={'title': 'Test Confusion Matrix'})

    current_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for data, target in train_iterator:
            current_step += 1
            label = target
            if torch.cuda.is_available():
                data, label = data.to('cuda'), label.to('cuda')
            # train model
            model.train()
            optimizer.zero_grad()
            classes = model(data)
            loss = sum([criterion(classes, label) for criterion in loss_criterion])
            loss.backward()
            optimizer.step()
            # save the metrics
            meter_loss.add(loss.detach().cpu().item())
            meter_accuracy.add(classes.detach().cpu(), target)
            meter_confusion.add(classes.detach().cpu(), target)

            if current_step % NUM_STEPS == 0:
                # print the information about train
                loss_logger.log(current_step // NUM_STEPS, meter_loss.value()[0], name='train')
                accuracy_logger.log(current_step // NUM_STEPS, meter_accuracy.value()[0], name='train')
                train_confusion_logger.log(meter_confusion.value())
                results['train_loss'].append(meter_loss.value()[0])
                results['train_accuracy'].append(meter_accuracy.value()[0])
                print('[Step %d] Training Loss: %.4f Accuracy: %.2f%%' % (
                    current_step // NUM_STEPS, meter_loss.value()[0], meter_accuracy.value()[0]))
                reset_meters()

                # test model periodically
                model.eval()
                with torch.no_grad():
                    for data, target in test_iterator:
                        label = target
                        if torch.cuda.is_available():
                            data, label = data.to('cuda'), label.to('cuda')
                        classes = model(data)
                        loss = sum([criterion(classes, label) for criterion in loss_criterion])
                        # save the metrics
                        meter_loss.add(loss.detach().cpu().item())
                        meter_accuracy.add(classes.detach().cpu(), target)
                        meter_confusion.add(classes.detach().cpu(), target)
                    # print the information about test
                    loss_logger.log(current_step // NUM_STEPS, meter_loss.value()[0], name='test')
                    accuracy_logger.log(current_step // NUM_STEPS, meter_accuracy.value()[0], name='test')
                    test_confusion_logger.log(meter_confusion.value())
                    results['test_loss'].append(meter_loss.value()[0])
                    results['test_accuracy'].append(meter_accuracy.value()[0])

                # save best model
                if meter_accuracy.value()[0] > best_acc:
                    best_acc = meter_accuracy.value()[0]
                    if FINE_GRAINED and DATA_TYPE in ['reuters', 'yelp', 'amazon']:
                        torch.save(model.state_dict(), 'epochs/{}_{}_{}_{}.pth'
                                   .format(DATA_TYPE + '_fine-grained', EMBEDDING_TYPE, CLASSIFIER_TYPE,
                                           str(TEXT_LENGTH)))
                    else:
                        torch.save(model.state_dict(), 'epochs/{}_{}_{}_{}.pth'
                                   .format(DATA_TYPE, EMBEDDING_TYPE, CLASSIFIER_TYPE, str(TEXT_LENGTH)))
                print('[Step %d] Testing Loss: %.4f Accuracy: %.2f%% Best Accuracy: %.2f%%' % (
                    current_step // NUM_STEPS, meter_loss.value()[0], meter_accuracy.value()[0], best_acc))
                reset_meters()

                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, current_step // NUM_STEPS + 1))
                if FINE_GRAINED and DATA_TYPE in ['reuters', 'yelp', 'amazon']:
                    data_frame.to_csv('statistics/{}_{}_{}_results.csv'.format(
                        DATA_TYPE + '_fine-grained', EMBEDDING_TYPE, CLASSIFIER_TYPE), index_label='step')
                else:
                    data_frame.to_csv('statistics/{}_{}_{}_results.csv'.format(
                        DATA_TYPE, EMBEDDING_TYPE, CLASSIFIER_TYPE), index_label='step')
        lr_scheduler.step(epoch)
