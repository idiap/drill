#    Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
#    Edited by Nikolaos Pappas <nikolaos.pappas@idiap.ch>.
#
#    This file is part of drill which builds over awd-lstm-lm codebase
#    (https://github.com/salesforce/awd-lstm-lm).
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

import os
import argparse
import time
import math
import numpy as np
np.random.seed(331)
import torch
import torch.nn as nn
import tables
import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--joint_emb', type=int, default=None,
                    help='whether to use joint embedding or not')
parser.add_argument('--joint_emb_depth', type=int, default=1,
                    help='depth of the joint embedding')
parser.add_argument('--joint_dropout', type=float, default=0.2,
                    help='dropout for joint embedding layers')
parser.add_argument('--joint_emb_dense', action='store_true', default=False,
                    help='add residuals to all previous joint embedding projections')
parser.add_argument('--joint_emb_activation', type=str, default='Sigmoid',
                    help='activation function for the joint embedding layers')
parser.add_argument('--joint_emb_gating', action="store_true",
                    help='gating instead of residual connection')
parser.add_argument('--joint_emb_dual', action='store_true', default=False,
                    help='whether to use projection on both outputs and weights or not (in the latter case only the weights projection is used)')
parser.add_argument('--joint_locked_dropout', action='store_true', default=False,
                    help='whether to use locked dropout or not for the joint space')
parser.add_argument('--joint_noresid', action='store_true', default=False,
                    help='disable residual connections')
parser.add_argument('--joint_residual_prev', action='store_true', default=False,
                    help='whether to use residual connection to previous layer')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', action="store_true", default=False,
                    help='resume or not')
parser.add_argument('--fullsoftmax', action='store_true',
                    help='whether to use full softmax or not')


args = parser.parse_args()

if args.fullsoftmax:
    args.tied = False

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

def logging(s, print_=True, log_=True):
    print(s)
    if log_:
        with open(os.path.join(args.save+'/log-finetuned.txt'), 'a+') as f_log:
            f_log.write(str(s) + '\n')


if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################


corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
logging('Args: %s'% args)
logging('Model total parameters: %d'  % total_params)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def store_datadist(data_source, model, batch_size=10, fname='datamatrix.h5'):
    """
       Store the log-probability matrix for a given method.
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    hidden = model.init_hidden(batch_size)

    # Initialize a data matrix structure which can be stored directly to the disk.
    f = tables.open_file(filename, mode='w')
    atom = tables.Float64Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 10000))

    # Add a row sequentially to the matrix for each different context.
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden)
        pred_targets = torch.mm(output, weight.t()) + bias
        hidden = repackage_hidden(hidden)
        datadist = nn.LogSoftmax()(pred_targets)
        array_c.append(datadist.detach().cpu().numpy())

    # Close file.
    f.close()

def store_word_cediff(data_source, model, batch_size=10, fname='out'):
    """
       Store the cross-entropy loss per word in the vocabulary.
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    hidden = model.init_hidden(batch_size)

    # Initialize vocabulary structure to store the crossentropy losses.
    vocab, words = {}, corpus.dictionary.idx2word

    # Add the loss per word in the vocabulary structure for each different context.
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden)
        pred_targets = torch.mm(output, weight.t()) + bias
        for j, target in enumerate(targets):
            target_loss = criterion(pred_targets[j:j+1], targets[j:j+1]).data
            word = words[target.tolist()]
            if word in vocab:
                vocab[word].append(target_loss.tolist())
            else:
                vocab[word] = [target_loss.tolist()]
        hidden = repackage_hidden(hidden)

    # Store the vocabulary to the disk.
    pickle.dump(vocab, open(fname+'.pkl','wb'))

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    if args.model == 'QRNN': model.reset()
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden)
        pred_targets = torch.mm(output, weight.t()) + bias
        total_loss += len(data) * criterion(pred_targets, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        model.use_dropout = True
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, weight, bias, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)

        pred_targets = torch.mm(output, weight.t()) + bias
        raw_loss = criterion(pred_targets, targets)

        loss = raw_loss
        # Activation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


# Load the best saved model.
with open(args.save+"/model.pt", 'rb') as f:
    model = torch.load(f)
    model.use_dropout = True
# Loop over epochs.
lr = args.lr
stored_loss = evaluate(val_data)
best_val_loss = []
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            logging('-' * 89)

            if val_loss2 < stored_loss:
                with open(args.save+'/model-finetuned.pt', 'wb') as f:
                    torch.save(model, f)
                logging('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        if (len(best_val_loss)>args.nonmono and val_loss2 > min(best_val_loss[:-args.nonmono])):
            logging('Done!')
            import sys
            sys.exit(1)
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            #optimizer.param_groups[0]['lr'] /= 2.
        best_val_loss.append(val_loss2)

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
with open(args.save+'/model-finetuned.pt', 'rb') as f:
    model = torch.load(f)#, map_location='cpu'

# Run on test data.
test_loss = evaluate(test_data)
# Store data distribution
#test_loss = evaluate_store_datadist(test_data, model, test_batch_size)
# Store cross entropy loss per word

logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)
