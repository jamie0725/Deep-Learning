# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, dropout,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        # save for forward
        self.seq_length = seq_length

        # embedding
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                      embedding_dim=lstm_num_hidden)

        # layers
        self.model = nn.LSTM(input_size=lstm_num_hidden, 
                             hidden_size=lstm_num_hidden,
                             num_layers=lstm_num_layers, 
                             bias=True,
                             dropout=dropout)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

        self.h_p = None

        self.to(device)


    def forward(self, x, h, c):
        embed = self.embedding(x)
        model, (h_n, c_n) = self.model(embed, (h, c))
        out = self.linear(model)
        return out, h_n, c_n
