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

import numpy as np

import torch

def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = -1

  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X

if __name__ == '__main__':
  V = 50
  h = 4
  bptt = 10
  batch_size = 2

  embed = torch.nn.Embedding(V, h)

  words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
  words = torch.LongTensor(words)

  origX = embed(words)
  X = embedded_dropout(embed, words)

  print(origX)
  print(X)
