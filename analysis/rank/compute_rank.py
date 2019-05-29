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

import sys
import numpy
import tables

filename = sys.argv[1]

# The log-prob matrix can be obtained by running the store_datadist() 
# function in finetune.py with the model of your preference.
print ("Loading matrix...")
f = tables.open_file(filename, mode='r')

print ("Loading rows into memory...")
datamatrix = f.root.data[:,:]
print (datamatrix.shape)
print (datamatrix[:5,:])

print ("Computing rank...")
print (numpy.linalg.matrix_rank(datamatrix, tol=True))

