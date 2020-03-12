from __future__ import print_function

import json
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
import os
# import utils_ctc
import crnn
import utils_ctc
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', default='../data/images/train_label.txt',help='path to training dataset')
parser.add_argument('--valid_root', default='../data/images/valid_label.txt', help='path to testing dataset')
parser.add_argument('--train_image_root', default='../data/images/train',help='')
parser.add_argument('--valid_image_root', default='../data/images/valid',help='')
parser.add_argument('--alphabet', default='../data/images/alphabet.txt', help='')
parser.add_argument('--direction', type=str, default='vertical', help='')
parser.add_argument('--std_mean_file', type=str, default='../data/images/desc/mean_std.json', help='')
parser.add_argument('--set_std_mean', action='store_true', help='')
parser.add_argument('--std', type=float, default=0.5, help='')
parser.add_argument('--mean', type=float, default=0.5, help='')
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=160, help='the width of the input image to network')
parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', default=False,help='enables cuda')
parser.add_argument('--opt', default='adam', help='select optimizer')
parser.add_argument('--nc', type=int, default=1, help='')
parser.add_argument('--expr', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--displayTrain', type=bool, default=False, help='Interval to be displayed')
# parser.add_argument('--testInterval', type=int, default=1, help='Interval to be displayed')
# parser.add_argument('--validInterval', type=int, default=1, help='Interval to be displayed')
# parser.add_argument('--saveInterval', type=int, default=1, help='Interval to save model')
parser.add_argument('--n_valid_disp', type=int, default=20, help='')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--best_acc', type=float, default=0.5, help='')
parser.add_argument('--keep_ratio', action='store_true', default=False,help='whether to keep ratio for image resize')
parser.add_argument('--mean_std_file', type=str, default='../data/images/desc/mean_std.json', help='')
arg = parser.parse_args()


# custom weights initialization called on crnn
from dataset_ctc import Dataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(crnn, valid_loader, criterion, epoch, max_i=1000):

    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    i = 0
    n_correct = 0
    loss_avg = utils_ctc.averager()

    for i_batch, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        # label = utils.get_batch_label(valid_dataset, index)
        preds = crnn(images)
        batch_size = images.size(0)
        # index = np.array(index.data.numpy())
        text, length = converter.encode(labels)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                n_correct += 1

        if (i_batch+1)%arg.displayInterval == 0:
            print('[%d/%d][%d/%d]' %
                      (epoch, arg.nepoch, i_batch, len(valid_loader)))

        if i_batch == max_i:
            break
        i+=1
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:arg.n_valid_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(i * arg.batch_size)
    accuracy = n_correct / float(i * arg.batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy

def train(crnn, train_loader, criterion, epoch):

    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    loss_avg = utils_ctc.averager()
    for i_batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        # label = utils.get_batch_label(train_dataset, index)
        preds = crnn(images)
        batch_size = images.size(0)
        # index = np.array(index.data.numpy())
        text, length = converter.encode(labels)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        # print(preds.shape, text.shape, preds_size.shape, length.shape)
        # torch.Size([41, 16, 6736]) torch.Size([160]) torch.Size([16]) torch.Size([16])

        cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        if (i_batch+1) % arg.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, arg.nepoch, i_batch, len(train_loader), loss_avg.val()))
            plot.add_loss(loss_avg.val())
            loss_avg.reset()

            if arg.displayTrain:
                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:arg.n_valid_disp]
                for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
                    print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))


def main(crnn, train_loader, valid_loader, criterion, optimizer):

    crnn = crnn.to(device)
    criterion = criterion.to(device)
    epoch = 0
    best_accuracy = 0.5
    while epoch < arg.nepoch:
        train(crnn, train_loader, criterion, epoch)
        ## max_i: cut down the consuming time of testing, if you'd like to validate on the whole testset, please set it to len(val_loader)
        accuracy = val(crnn, valid_loader, criterion, epoch, max_i=1000)
        for p in crnn.parameters():
            p.requires_grad = True
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format('./expr', epoch, accuracy))
            torch.save(crnn.state_dict(), '{0}/crnn_best.pth'.format('./expr'))
        print("best accuracy update: {0}".format(accuracy))
        epoch+=1
        plot.add_acc(accuracy,epoch)
    plot.show()

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

if __name__ == '__main__':

    plot= utils.Plot(arg.nepoch, fname='expr/loss.png')

    manualSeed=10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # alphabet = alphabet = utils.to_alphabet("H:/DL-DATASET/BaiduTextR/train.list")

    # store model path
    if not os.path.exists(arg.expr):
        os.mkdir(arg.expr)
    if arg.set_std_mean:
        std_mean=(arg.std,arg.mean)
    else:
        std_mean=json.load(open(arg.std_mean_file))
        std_mean=(std_mean['L']['std'],std_mean['L']['mean'])
    print('std_mean',std_mean)
    # read train set
    train_dataset = Dataset(
        arg.train_image_root,
        arg.train_root,
        std_mean=std_mean,
        imgW=arg.imgW,
        imgH=arg.imgH,
        direction=arg.direction
    )
    valid_dataset = Dataset(
        arg.valid_image_root,
        arg.valid_root,
        std_mean=std_mean,
        imgW=arg.imgW,
        imgH=arg.imgH,
        direction=arg.direction
    )

    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers, drop_last=True)
    with open(arg.alphabet,encoding='utf-8') as f:
        alphabet=f.read()
    converter = utils_ctc.strLabelConverter(alphabet)
    nclass = len(alphabet) + 1
    nc = 1
    nh=256
    criterion = torch.nn.CTCLoss(reduction='sum')
    # criterion = CTCLoss()

    # cnn and rnn
    crnn = crnn.CRNN(arg.imgH, nc, nclass, nh)
    print(crnn)
    crnn.apply(weights_init)
    # if params.crnn != '':
    #     print('loading pretrained model from %s' % params.crnn)
    #     crnn.load_state_dict(torch.load(params.crnn))

    # loss averager
    # loss_avg = utils.averager()

    # setup optimizer
    # optimizer = optim.Adam(crnn.parameters(), lr=arg.lr,
    #                            betas=(0.99, 0.999))

    # optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)

    optimizer = optim.RMSprop(crnn.parameters(), lr=arg.lr)

    crnn.register_backward_hook(backward_hook)
    main(crnn, train_loader, valid_loader, criterion, optimizer)
