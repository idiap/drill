#    Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
#    Edited by Nikolaos Pappas <nikolaos.pappas@idiap.ch>.
#
#    This file is part of drill which builds over OpenNMT-py codebase
#    (https://github.com/OpenNMT/OpenNMT-py).
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
import math
import numpy
import onmt
import random
import pickle

import torch.nn as nn
from torch.autograd import Variable
from onmt.modules.locked_dropout import LockedDropout

class DrillGenerator(nn.Module):
    def __init__(self, opt, decoder, gen_func):
        super(DrillGenerator, self).__init__()
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        self.decoder = decoder
        self.gen_func = gen_func
        self.opt = opt
        self.ninp = decoder.embeddings.emb_luts[0].weight.shape[1]
        self.ntoken = decoder.embeddings.emb_luts[0].weight.shape[0]

        self.dropjoint = nn.Dropout(self.opt.joint_dropout if self.use_dropout else 0)

        # Define the first layer of the label encoder network
        self.joint_encoder_proj_0 = nn.Sequential(nn.Linear(self.ninp, self.ninp if self.opt.joint_emb_depth < 2 else self.opt.joint_emb, bias=True), eval("nn.%s()" % self.opt.joint_emb_activation))
        self.encoder_projs = [self.joint_encoder_proj_0]

        # (Optional) Define the first layer of the input encoder network
        if self.opt.joint_emb_dual:
            self.joint_hidden_proj_0 = nn.Sequential(nn.Linear(self.ninp, self.opt.joint_emb, bias=True), eval("nn.%s()" % self.opt.joint_emb_activation))
            self.hidden_projs = [self.joint_hidden_proj_0]

       # Define the subsequent layers for the input and label encoder networks
       for i in range(self.opt.joint_emb_depth - 1):
            exec("self.joint_encoder_proj_%d = nn.Sequential(nn.Linear(self.opt.joint_emb, self.ninp if i == self.opt.joint_emb_depth - 2 else self.opt.joint_emb, bias=True), nn.%s())" % (i+1, self.opt.joint_emb_activation))
            if self.opt.joint_emb_dual:
                exec("self.joint_hidden_proj_%d = nn.Sequential(nn.Linear(self.opt.joint_emb, self.ninp if i == self.opt.joint_emb_depth - 2 else self.joint_emb, bias=True), nn.%s())" % (i+1, self.opt.joint_emb_activation))
                self.hidden_projs.append(eval("self.joint_hidden_proj_%d" % (i+1)))
            self.encoder_projs.append(eval("self.joint_encoder_proj_%d" % (i+1)))
        self.bias = torch.nn.Linear(self.ntoken, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if hasattr(self, 'encoder_projs'):
            for i in range(self.opt.joint_emb_depth):
                if self.opt.joint_emb_dual:
                    self.hidden_projs[i][0].weight.data.uniform_(-initrange, initrange)
                self.encoder_projs[i][0].weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'bias'):
            self.bias.weight.data.fill_(0)

    def forward(self, hidden):
        hidden, weight = self.apply_drill(hidden, self.decoder.embeddings.emb_luts[0].weight)
        v_out = torch.mm(hidden, weight.t()) + self.bias.weight.view(-1)
        return self.gen_func(v_out)


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
