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

import torch
import torch.nn as nn
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, joint_emb=None, joint_emb_depth=0, joint_emb_dense=False, joint_emb_dual=True,  joint_dropout=0.2,  joint_emb_activation='Sigmoid',  joint_locked_dropout=False, joint_residual_prev=False, joint_noresid=False):
        super(RNNModel, self).__init__()
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti if self.use_dropout else 0)
        self.hdrop = nn.Dropout(dropouth if self.use_dropout else 0)
        self.drop = nn.Dropout(dropout if self.use_dropout else 0)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights or (joint_emb is not None) else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop if self.use_dropout else 0) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop  if self.use_dropout else 0) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop if self.use_dropout else 0)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        if joint_emb is None:
            if tie_weights:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder = nn.Linear(ninp, ntoken)
                self.decoder.weight = self.encoder.weight
            else:
                self.decoder = nn.Linear(nhid, ntoken)
        else:
            self.dropjoint = nn.Dropout(joint_dropout if self.use_dropout else 0)

            # Define the first layer of the label encoder network
             if joint_emb_activation != "Linear":
                self.joint_encoder_proj_0 = nn.Sequential(nn.Linear(ninp, joint_emb, bias=True), eval("nn.%s()" % joint_emb_activation))
            else:
                self.joint_encoder_proj_0 = nn.Sequential(nn.Linear(ninp, joint_emb, bias=True))
            self.encoder_projs = [self.joint_encoder_proj_0]

            # (Optional) Define the first layer of the input encoder network
            if joint_emb_dual:
                if joint_emb_activation != "Linear":
                    self.joint_hidden_proj_0 = nn.Sequential(nn.Linear(ninp, joint_emb, bias=True), eval("nn.%s()" % joint_emb_activation))
                else:
                    self.joint_hidden_proj_0 = nn.Sequential(nn.Linear(ninp, joint_emb, bias=True))
                self.hidden_projs = [self.joint_hidden_proj_0]
            # Define the subsequent layers for the input and label encoder networks
            for i in range(joint_emb_depth - 1):
                if joint_emb_activation != "Linear":
                    exec("self.joint_encoder_proj_%d = nn.Sequential(nn.Linear(joint_emb, joint_emb, bias=True), nn.%s())" % (i+1, joint_emb_activation))
                else:
                    exec("self.joint_encoder_proj_%d = nn.Sequential(nn.Linear(joint_emb, joint_emb, bias=True)")
                if joint_emb_dual:
                    if joint_emb_activation != "Linear":
                        exec("self.joint_hidden_proj_%d = nn.Sequential(nn.Linear(joint_emb, joint_emb, bias=True), nn.%s())" % (i+1, joint_emb_activation))
                    else:
                        exec("self.joint_hidden_proj_%d = nn.Sequential(nn.Linear(joint_emb, joint_emb, bias=True)")
                    self.hidden_projs.append(eval("self.joint_hidden_proj_%d" % (i+1)))
                self.encoder_projs.append(eval("self.joint_encoder_proj_%d" % (i+1)))
            self.bias = torch.nn.Linear(ntoken, 1, bias=False)
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.tie_weights = tie_weights
        self.joint_emb = joint_emb
        self.joint_emb_depth = joint_emb_depth
        self.joint_emb_dual = joint_emb_dual
        self.joint_dropout = joint_dropout
        self.joint_locked_dropout = joint_locked_dropout
        self.joint_residual_prev = joint_residual_prev
        self.joint_noresid = joint_noresid
        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'decoder'):
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'encoder_projs'):
            for i in range(self.joint_emb_depth):
                if self.joint_emb_dual:
                    self.hidden_projs[i][0].weight.data.uniform_(-initrange, initrange)
                self.encoder_projs[i][0].weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'bias'):
            self.bias.weight.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights or (self.joint_emb is not None) else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights or (self.joint_emb is not None) else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights or (self.joint_emb is not None) else self.nhid)).zero_()
                    for l in range(self.nlayers)]

    def forward(self, input, hidden, return_h=False, train=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training and self.use_dropout else 0)
        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)

            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)

        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout if self.use_dropout else 0)
        outputs.append(output)
        result = output.view(output.size(0)*output.size(1), output.size(2))
        weight = self.encoder.weight if self.tie_weights or self.joint_emb is not None else self.decoder.weight
        bias = self.decoder.bias if self.tie_weights or self.joint_emb is None else self.bias.weight

        if self.joint_emb is not None:
            result, weight = self.apply_drill(output, weight)

        if return_h:
            return result, weight, bias, hidden, raw_outputs, outputs
        return result, weight, bias, hidden

    def apply_drill(self, output, weight):
        origshape = output.shape
        output = output.view(-1, self.ninp)
        prevs_hidden_out = [output]
        prevs_encoder_out = [weight]
        for i in range(self.joint_emb_depth):
            # Forward through the label encoder network
            if self.joint_locked_dropout:
                cur_weight = self.lockdrop(prevs_encoder_out[i].view(self.ntoken, 1, self.ninp), self.joint_dropout if self.use_dropout else 0).view(self.ntoken, self.ninp)
            else:
                cur_weight = self.dropjoint(prevs_encoder_out[i]) if self.use_dropout else prevs_encoder_out[i]
            cur_weight_proj = self.encoder_projs[i](cur_weight)

            if i > 0 and self.joint_residual_prev:
                cur_weight_proj = cur_weight_proj + prevs_encoder_out[i]

            if not self.joint_noresid:
                cur_weight_proj = cur_weight_proj + weight

            # (Optional) Forward through the input encoder network
            if self.joint_emb_dual:
                if self.joint_locked_dropout:
                    cur_output = self.lockdrop(prevs_hidden_out[i].view(origshape), self.joint_dropout if self.use_dropout else 0).view(-1, self.ninp) if i > 0 else prevs_hidden_out[i]
                else:
                    cur_output = self.dropjoint(prevs_hidden_out[i]) if i > 0 and  self.use_dropout else prevs_hidden_out[i]
                cur_output_proj = self.hidden_projs[i](cur_output)

                if i > 0 and self.joint_residual_prev:
                    cur_output_proj = cur_output_proj + prevs_hidden_out[i]

               if not self.joint_noresid:
                    cur_output_proj = cur_output_proj + output

                prevs_hidden_out.append(cur_output_proj)
            prevs_encoder_out.append(cur_weight_proj)
        h_final = prevs_hidden_out[-1].view(-1, self.ninp)
        E_final = prevs_encoder_out[-1]
        return h_final, E_final
