################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # save for forward
        self.seq_length = seq_length

        # weights
        self.W_gx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_gh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.W_ix = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_ih = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.W_fx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_fh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.W_ox = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.W_oh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.W_ph = nn.Parameter(torch.randn(num_classes, num_hidden))

        # bias
        self.b_g = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_i = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_f = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_o = nn.Parameter(torch.zeros(num_hidden, 1))
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        # activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # initial hidden state
        self.h_i = nn.Parameter(torch.zeros(num_hidden, batch_size), requires_grad=False)
        self.c_t = nn.Parameter(torch.zeros(num_hidden, batch_size), requires_grad=False)

    def forward(self, x):
        h_t = self.h_i
        c_t = self.c_t
        for i in range(self.seq_length):
            g_t = self.tanh(self.W_gx @ x[:,i].view(1, -1) + self.W_gh @ h_t + self.b_g)
            i_t = self.sigmoid(self.W_ix @ x[:,i].view(1, -1) + self.W_ih @ h_t + self.b_i)
            f_t = self.sigmoid(self.W_fx @ x[:,i].view(1, -1) + self.W_fh @ h_t + self.b_f)
            o_t = self.sigmoid(self.W_ox @ x[:,i].view(1, -1) + self.W_oh @ h_t + self.b_o)
            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t
        out = self.W_ph @ h_t + self.b_p
        out = out.t()
        return out