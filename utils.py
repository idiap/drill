#    Copyright (c) awd-lstm-lm, https://github.com/salesforce/awd-lstm-lm
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

import torch
import os, shutil

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, criterion, optimizer, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(criterion, os.path.join(path, 'finetune_criterion.pt'))
        torch.save(optimizer, os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(criterion, os.path.join(path, 'criterion.pt'))
        torch.save(optimizer, os.path.join(path, 'optimizer.pt'))
