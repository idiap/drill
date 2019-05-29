#    Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
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

import pickle
import numpy as np
import codecs
import sys
from pylab import *

def cediff_per_bin(dataset, method):
    """
        Function to compute the mean relative crossentropy loss difference
        for different word frequency bins between two models. To calculate 
        the cross-entropy loss difference per word and store them into pkl 
        files by running the store_word_cediff() function in finetune.py 
        with the model of your preference.
    """
    vocab = pickle.load(open(dataset+'/'+dataset+'-vocab.pkl', 'rb'))
    ours_vocab = pickle.load(open(dataset+'/'+dataset+'-ours.pkl', 'rb'))
    base_vocab = pickle.load(open(dataset+'/'+dataset+'-'+method+'.pkl', 'rb'))
    s = [(k, vocab[k]) for k in sorted(vocab, key=vocab.get, reverse=True)]
    ranges    = [(1, 50), (50,100), (100, 250), (250,500), (500, 1000), (1000, 2500), (2500, 5000), (5000,  1000000)]

    for (minval, maxval) in ranges:
        total, count = [], 0
        for word, freq in s:
            if word in base_vocab and freq < maxval and freq >= minval:
                base_val = np.array(base_vocab[word])
                ours_val = np.array(ours_vocab[word])
                base = np.sum(base_val)
                ours = np.sum(ours_val)
                res = ((base - ours)/ours)*100
                total.append(res)
                count += 1
        tot = np.mean(np.array(total))
        print (tot, len(total))

if __name__ == "__main__":
    dataset = sys.argv[1]
    method = sys.argv[2]
    cediff_per_bin(dataset, method)

