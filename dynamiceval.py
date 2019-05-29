#    Copyright (c) dynamic-evaluation, https://github.com/benkrause/dynamic-evaluation
#
#    This file is part of drill which builds over dynamic-evaluation codebase
#    (https://github.com/benkrause/dynamic-evaluation).
#
#    drill is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    drill is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with drill. If not, see http://www.gnu.org/licenses/

import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str,
                    help='name of model to eval')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu device ID (-1 for cpu)')
parser.add_argument('--val', action='store_true',
                    help='set for validation error, test by default')
parser.add_argument('--lamb', type=float, default=0.002,
                    help='decay parameter lambda')
parser.add_argument('--epsilon', type=float, default=0.00002,
                    help='stabilization parameter epsilon')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate eta')
parser.add_argument('--oldhyper', action='store_true',
                    help='Transforms hyperparameters, equivalent to running old version of code')
parser.add_argument('--grid', action='store_true',
                    help='grid search for best hyperparams')
parser.add_argument('--gridfast', action='store_true',
                    help='grid search with partial validation set')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size for gradient statistics')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence/truncation length')
parser.add_argument('--max_batches', type=int, default=-1,
                    help='maximum number of training batches for gradient statistics')
parser.add_argument('--QRNN', action='store_true',
                    help='Use if model is a QRNN')


args = parser.parse_args()

if args.gpu>=0:
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    #to run on cpu, model must have been trained on cpu
    args.cuda=False

model_name=args.model

print('loading')

corpus = data.Corpus(args.data)
eval_batch_size = 1
test_batch_size = 1



def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data
#######################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def gradstat():

    if args.QRNN:
        model.reset()

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0

    for param in model.parameters():
        param.MS = 0*param.data


    while i < train_data.size(0) - 1 - 1:
        seq_len = args.bptt
        model.train()
        model.use_dropout = False
        data, targets = get_batch(train_data, i)

        hidden = repackage_hidden(hidden)

        model.zero_grad()
        #assumes model has atleast 2 returns, and first is output and second is hidden
        #returns = model(data, hidden)
        #output = returns[0]
        #hidden = returns[1]
        output, weight, bias, hidden = model(data, hidden)
        output =  torch.mm(output, weight.t()) + bias


        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        for param in model.parameters():
            param.MS = param.MS + param.grad.data*param.grad.data

        total_loss += loss.data

        batch += 1


        i += seq_len
        if args.max_batches>0:
            if batch>= args.max_batches:
                break
    gsum = 0
    count = 0

    for param in model.parameters():

        param.MS = torch.sqrt(param.MS/batch)

        gsum+=torch.mean(param.MS)
        count+=1
    gsum/=count
    if args.oldhyper:
        args.lamb /=count
        args.lr /=math.sqrt(batch)
        args.epsilon /=math.sqrt(batch)
        print("transformed lambda: " + str(args.lamb))
        print("transformed lr: " + str(args.lr))
        print("transformed epsilon: " + str(args.epsilon))


    for param in model.parameters():
        param.decrate = param.MS/gsum
        param.data0 = 1*param.data

def evaluate():

    if args.QRNN:
       model.reset()
    #clips decay rates at 1/lamb
    #otherwise scaled decay rates can be greater than 1
    #would cause decay updates to overshoot
    for param in model.parameters():
        if args.cuda:
            decratenp = param.decrate.cpu().numpy()
            ind = np.nonzero(decratenp>(1/lamb))
            decratenp[ind] = (1/lamb)
            param.decrate = torch.from_numpy(decratenp).type(torch.cuda.FloatTensor)

        else:
            decratenp = param.decrate.numpy()
            ind = np.nonzero(decratenp>(1/lamb))
            decratenp[ind] = (1/lamb)
            param.decrate = torch.from_numpy(decratenp).type(torch.FloatTensor)


    total_loss = 0

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    last = False
    seq_len= args.bptt
    seq_len0 = seq_len
    #loops through data
    while i < eval_data.size(0) - 1 - 1:

        model.train()
        model.use_dropout = False

        #gets last chunk of seqlence if seqlen doesn't divide full sequence cleanly
        if (i+seq_len)>=eval_data.size(0):
            if last:
                break
            seq_len = eval_data.size(0)-i-1
            last = True

        data, targets = get_batch(eval_data,i)

        hidden = repackage_hidden(hidden)

        model.zero_grad()

        #assumes model has atleast 2 returns, and first is output and second is hidden
        output, weight, bias, hidden = model(data, hidden)
        output =  torch.mm(output, weight.t()) + bias
        loss = criterion(output.view(-1, ntokens), targets)

        #compute gradient on sequence segment loss
        loss.backward()

        #update rule
        for param in model.parameters():
            dW = lamb*param.decrate*(param.data0-param.data)-lr*param.grad.data/(param.MS+epsilon)
            param.data+=dW

        #seq_len/seq_len0 will be 1 except for last sequence
        #for last sequence, we downweight if sequence is shorter
        total_loss += (seq_len/seq_len0)*loss.data
        batch += (seq_len/seq_len0)
        i += seq_len

    #since entropy of first token was never measured
    #can conservatively measure with uniform distribution
    #makes very little difference, usually < 0.01 perplexity point
    #total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    #batch+=(1/seq_len0)

    perp = torch.exp(total_loss/batch)
    if args.cuda:
        return perp.cpu().numpy()
    else:
        return perp.numpy()

#load model
with open(model_name, 'rb') as f:
    model = torch.load(f)

ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)

if args.val== True:
    eval_data= val_data
else:
    eval_data=test_data
train_data = batchify(corpus.train, args.batch_size)

print('collecting gradient statistics')
#collect gradient statistics on training data
gradstat()

lr = args.lr
lamb = args.lamb
epsilon = args.epsilon

#change batch size to 1 for dynamic eval
args.batch_size=1
if not(args.grid or args.gridfast):
    print('running dynamic evaluation')
    #apply dynamic evaluation
    loss = evaluate()
    print('perplexity loss: ' + str(loss))
else:
    vbest = 99999999
    lambbest = lamb
    lrbest = lr

    if args.gridfast:
        eval_data = val_data[:30000]
    else:
        eval_data = val_data
    print('tuning hyperparameters')

    #hyperparameter values to be searched
    lrlist = [0.00003,0.00004,0.00005,0.00006,0.00007,0.0001]
    lamblist = [0.001,0.002,0.003,0.005]

    #rescale values if sequence segment length is changed
    lrlist = [x*(args.bptt/5.0) for x in lrlist]
    lamblist = [x*(args.bptt/5.0) for x in lamblist]

    for i in range(0,len(lamblist)):
        for j in range(0,len(lrlist)):
            lamb = lamblist[i]
            lr = lrlist[j]
            loss = evaluate()
            loss = loss
            if loss<vbest:
                lambbest = lamb
                lrbest = lr
                vbest = loss
            for param in model.parameters():
                param.data = 1*param.data0
    print('best hyperparams: lr = ' + str(lrbest) + ' lamb = '+ str(lambbest))
    print('getting validation and test error')
    eval_data = val_data
    lamb = lambbest
    lr = lrbest
    vloss = evaluate()
    for param in model.parameters():
        param.data = 1*param.data0

    eval_data = test_data
    lamb = lambbest
    lr = lrbest
    tloss = evaluate()
    print('validation perplexity loss: ' + str(vloss))
    print('test perplexity loss: ' + str(tloss))
